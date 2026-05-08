"""Limpeza, normalização, chunking e extração heurística de seções jurídicas."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Final

_MONEY_PLACEHOLDER: Final[str] = "\uE000MONEY\uE001"
_DATE_PLACEHOLDER: Final[str] = "\uE000DATE\uE001"
_ARTICLE_PLACEHOLDER: Final[str] = "\uE000ART\uE001"
_MONTH_NAME_PATTERN: Final[str] = (
    r"janeiro|fevereiro|mar[cç]o|abril|maio|junho|julho|agosto|"
    r"setembro|outubro|novembro|dezembro"
)

_MONEY_PATTERN = re.compile(
    r"(?<!\w)(?:R\$\s*)?\d{1,3}(?:\.\d{3})*(?:,\d{2})\b|"
    r"\bR\$\s*\d{1,3}(?:\.\d{3})*(?:,\d{2})\b|"
    r"\bR\$\s*\d+(?:,\d{2})?\b",
    re.IGNORECASE,
)
_DATE_PATTERN = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
    rf"\b\d{{1,2}}\s+de\s+(?:{_MONTH_NAME_PATTERN})\s+de\s+\d{{4}}\b|"
    r"\b\d{1,2}[/-]\d{4}\b|"
    rf"\b(?:{_MONTH_NAME_PATTERN})\s+de\s+\d{{4}}\b|"
    r"\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_ARTICLE_PATTERN = re.compile(
    r"(?:^|\s)(?:Art(?:igo)?\.?\s*\d+[\s.,\-–—]*[ºª]?|"
    r"Cláusula\s+(?:[IVXLCDM]+|\d+)[ºª]?|"
    r"§\s*\d+[\s.,\-–—]*)",
    re.IGNORECASE | re.MULTILINE,
)

_PARTES_LINE = re.compile(
    r"(?im)^\s*(CONTRATANTE|CONTRATADA|CONTRATADO|OUTORGANTE|OUTORGADO|"
    r"VENDEDOR|COMPRADOR|LOCADOR|LOCATÁRIO|LOCATARIO|"
    r"REPRESENTANTE|PARTE\s+[AB])\s*:\s*(.+)$",
)
_CLAUSE_HEAD = re.compile(
    r"(?im)^\s*(?:(?:CLÁUSULA|CLAUSULA)\s+(?:[IVXLCDM]+|\d+)[ºª]?\s*"
    r"[.:)\-–—]?\s*(.{0,800})|"
    r"(?:^|\n)\s*(?:\d{1,3}|[IVXLCDM]+)\s*[-–—.)]\s+(.{0,800}))",
)


class TextPreprocessor:
    """Pré-processamento de texto para pipelines de NLP jurídico."""

    def clean(self, text: str) -> str:
        if not text:
            return ""
        preserved_money: list[str] = []
        preserved_dates: list[str] = []
        preserved_articles: list[str] = []

        def _shield_money(m: re.Match[str]) -> str:
            preserved_money.append(m.group(0))
            return _MONEY_PLACEHOLDER

        def _shield_date(m: re.Match[str]) -> str:
            preserved_dates.append(m.group(0))
            return _DATE_PLACEHOLDER

        def _shield_art(m: re.Match[str]) -> str:
            preserved_articles.append(m.group(0))
            return _ARTICLE_PLACEHOLDER

        shielded = text
        shielded = _MONEY_PATTERN.sub(_shield_money, shielded)
        shielded = _DATE_PATTERN.sub(_shield_date, shielded)
        shielded = _ARTICLE_PATTERN.sub(_shield_art, shielded)

        shielded = self._strip_repeated_headers_footers(shielded)
        shielded = self._normalize_unicode_and_noise(shielded)
        shielded = self._collapse_whitespace(shielded)

        out = shielded
        for val in preserved_money:
            out = out.replace(_MONEY_PLACEHOLDER, val, 1)
        for val in preserved_dates:
            out = out.replace(_DATE_PLACEHOLDER, val, 1)
        for val in preserved_articles:
            out = out.replace(_ARTICLE_PLACEHOLDER, val, 1)

        return out.strip()

    def _strip_repeated_headers_footers(self, text: str) -> str:
        lines = text.splitlines()
        if len(lines) < 6:
            return text
        stripped = [ln.strip() for ln in lines]
        counts = Counter(s for s in stripped if s)
        to_drop: set[str] = set()
        for line, c in counts.items():
            if c >= 3 and len(line) <= 140:
                to_drop.add(line)
        if not to_drop:
            return text
        kept: list[str] = []
        for raw, s in zip(lines, stripped):
            if s and s in to_drop:
                continue
            kept.append(raw)
        return "\n".join(kept)

    def _normalize_unicode_and_noise(self, text: str) -> str:
        nfkc = unicodedata.normalize("NFKC", text)
        nfkc = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", nfkc)
        nfkc = re.sub(r"\uFEFF", "", nfkc)
        nfkc = nfkc.replace("\r\n", "\n").replace("\r", "\n")
        nfkc = re.sub(r"[ \t]+", " ", nfkc)
        nfkc = re.sub(r"\n{3,}", "\n\n", nfkc)
        nfkc = re.sub(r"\s+([.,;:!?])", r"\1", nfkc)
        return nfkc

    def _collapse_whitespace(self, text: str) -> str:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        return "\n\n".join(parts)

    def split_into_chunks(
        self,
        text: str,
        max_tokens: int = 450,
        overlap: int = 50,
    ) -> list[str]:
        if max_tokens < 1:
            raise ValueError("max_tokens deve ser >= 1")
        if overlap < 0 or overlap >= max_tokens:
            raise ValueError("overlap deve estar em [0, max_tokens)")

        cleaned = self.clean(text)
        paragraphs = [[w for w in p.split()] for p in cleaned.split("\n\n") if p.strip()]
        if not paragraphs:
            return []

        chunks: list[str] = []
        buf: list[str] = []

        def flush_with_overlap() -> None:
            nonlocal buf
            if not buf:
                return
            chunks.append(" ".join(buf))
            if overlap > 0 and len(buf) > overlap:
                buf = buf[-overlap:]
            else:
                buf = []

        for para in paragraphs:
            idx = 0
            while idx < len(para):
                room = max_tokens - len(buf)
                if room == 0:
                    flush_with_overlap()
                    continue
                take = min(room, len(para) - idx)
                buf.extend(para[idx : idx + take])
                idx += take
                if len(buf) >= max_tokens:
                    flush_with_overlap()

        if buf:
            chunks.append(" ".join(buf))

        return chunks

    def extract_summary_focus_text(self, text: str, document_kind: str) -> str:
        """
        Trecho priorizado para sumarização: seções de petição ou partes/cláusulas
        em contrato; reduz ruído de cabeçalhos judiciários no encoder do T5.
        """
        cleaned = self.clean(text) if text else ""
        if not cleaned.strip():
            return ""

        if document_kind == "peticao_inicial":
            fatos = re.search(
                r"(?is)\bDOS\s+FATOS\b\s*[.:]?\s*(.{200,12000}?)(?=\n\s*\bDOS\s+|\n\s*\bDO\s+DIREITO\b|\Z)",
                cleaned,
            )
            direito = re.search(
                r"(?is)\bDO\s+DIREITO\b\s*[.:]?\s*(.{200,6000}?)(?=\n\s*\bDOS\s+PEDIDOS\b|\Z)",
                cleaned,
            )
            pedidos = re.search(
                r"(?is)\bDOS\s+PEDIDOS\b\s*[.:]?\s*(.{200,8000}?)(?=\n\s*\bDOS\s+|\Z)",
                cleaned,
            )
            parts: list[str] = []
            if fatos:
                parts.append(fatos.group(1).strip())
            if direito:
                parts.append(direito.group(1).strip())
            if pedidos:
                parts.append(pedidos.group(1).strip())
            if parts:
                return "\n\n".join(parts)[:12000]
            return cleaned[:12000]

        secs = self.extract_sections(cleaned)
        cl = "\n".join(secs.get("clausulas", [])[:15])
        pa = "\n".join(secs.get("partes", [])[:10])
        combo = (pa + "\n\n" + cl).strip()
        if len(combo) > 400:
            return combo[:12000]
        return cleaned[:12000]

    def extract_sections(self, text: str) -> dict[str, list[str]]:
        partes: list[str] = []
        clausulas: list[str] = []
        valores: list[str] = []
        datas: list[str] = []

        for m in _PARTES_LINE.finditer(text):
            rest = (m.group(2) or "").strip()
            if rest and len(rest) < 500:
                partes.append(rest)

        for m in _CLAUSE_HEAD.finditer(text):
            body = next((g for g in m.groups() if g), None)
            line = (body or m.group(0)).strip()
            if line:
                clausulas.append(line[:2000])

        for m in _MONEY_PATTERN.finditer(text):
            valores.append(m.group(0).strip())

        for m in _DATE_PATTERN.finditer(text):
            datas.append(m.group(0).strip())

        return {
            "partes": self._unique_preserve_order(partes),
            "clausulas": self._unique_preserve_order(clausulas),
            "valores": self._unique_preserve_order(valores),
            "datas": self._unique_preserve_order(datas),
        }

    def extract_clauses(self, text: str) -> list[dict]:
        """
        Extrai cláusulas numeradas com tipo inferido e texto.
        Retorna lista de {numero, tipo, texto}.
        """
        pattern = re.compile(
            r"(?im)^\s*(?:CL[AÁ]USULA\s+(?P<num1>[IVXLCDM]+|\d+)[ºª]?"
            r"(?:\s*[-–—:.]?\s*(?P<titulo1>[^\n]{0,80}))?|"
            r"(?P<num2>\d{1,2})\.\s+(?P<titulo2>[A-ZÁÉÍÓÚÂÊÔÃÕÇÜ][^\n]{5,80}))"
            r"\s*\n(?P<corpo>(?:(?!\n\s*(?:CL[AÁ]USULA|\d{1,2}\.)\s).){20,1500})",
            re.DOTALL,
        )
        keyword_tipo = [
            ("multa",        ["multa", "penalidade", "pena"]),
            ("rescisão",     ["rescis", "resolução", "término", "encerramento"]),
            ("pagamento",    ["pagamento", "remunera", "honorário", "valor", "preço"]),
            ("prazo",        ["prazo", "vigência", "duração", "vencimento"]),
            ("lgpd",         ["dado pessoal", "lgpd", "tratamento de dado", "privacidade"]),
            ("sigilo",       ["sigilo", "confidencialidade", "segredo"]),
            ("foro",         ["foro", "jurisdição", "competência"]),
            ("garantia",     ["garantia", "caução", "fiança"]),
            ("propriedade_intelectual", ["propriedade intelectual", "direito autoral", "patente", "marca"]),
            ("responsabilidade", ["responsabilidade", "indenização", "dano"]),
        ]

        results = []
        for m in pattern.finditer(text):
            num = (m.group("num1") or m.group("num2") or "").strip()
            titulo = (m.group("titulo1") or m.group("titulo2") or "").strip()
            corpo = (m.group("corpo") or "").strip()
            texto_completo = (titulo + " " + corpo).lower()

            tipo = "geral"
            for t, kws in keyword_tipo:
                if any(kw in texto_completo for kw in kws):
                    tipo = t
                    break

            results.append({
                "numero": num,
                "tipo": tipo,
                "titulo": titulo[:120] if titulo else "",
                "texto": corpo[:500],
            })

        return results[:30]

    def extract_timeline(self, text: str, dates: list[str]) -> list[dict]:
        """
        Monta linha do tempo extraindo contexto em torno de cada data.
        Retorna lista de {data, evento} ordenada cronologicamente.
        """
        MONTHS = {
            "janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4,
            "marco": 3,
            "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
            "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
        }

        def valid_date(year: int, month: int, day: int) -> bool:
            if year < 1 or month < 1 or month > 12 or day < 1 or day > 31:
                return False
            if month in {4, 6, 9, 11} and day > 30:
                return False
            if month == 2:
                leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
                return day <= (29 if leap else 28)
            return True

        def parse_date(s: str) -> tuple[int, int, int] | None:
            s = s.strip().lower()
            m = re.fullmatch(r"(\d{1,2})\s+de\s+([a-zçãõáéíóú]+)\s+de\s+(\d{4})", s)
            if m:
                day, month_name, year = int(m.group(1)), m.group(2), int(m.group(3))
                month = MONTHS.get(month_name, 0)
                if valid_date(year, month, day):
                    return (year, month, day)
            m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", s)
            if m:
                day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if year < 100:
                    year += 2000
                if valid_date(year, month, day):
                    return (year, month, day)
            m = re.fullmatch(r"(\d{1,2})[/-](\d{4})", s)
            if m:
                month, year = int(m.group(1)), int(m.group(2))
                if valid_date(year, month, 1):
                    return (year, month, 1)
            m = re.fullmatch(r"([a-zçãõáéíóú]+)\s+de\s+(\d{4})", s)
            if m:
                month, year = MONTHS.get(m.group(1), 0), int(m.group(2))
                if valid_date(year, month, 1):
                    return (year, month, 1)
            m = re.fullmatch(r"\d{4}", s)
            if m:
                return (int(s), 1, 1)
            return None

        events = []
        seen_dates: set[str] = set()
        for date_str in dates:
            if date_str in seen_dates:
                continue
            seen_dates.add(date_str)

            idx = text.find(date_str)
            if idx == -1:
                continue

            start = max(0, idx - 120)
            end = min(len(text), idx + len(date_str) + 180)
            context = text[start:end].strip()
            context = re.sub(r"\s+", " ", context)

            # Extrai verbo/ação próximo à data
            action_match = re.search(
                r"(?:foi|foram|é|houve|ocorreu|realizou|apresentou|"
                r"ajuizou|citou|julgou|condenou|determinou|sentenciou|"
                r"interpôs|proferiu|concedeu|indeferiu|deferiu)\b[^.]{0,80}",
                context, re.I
            )
            evento = action_match.group(0).strip() if action_match else context[:120]

            sort_key = parse_date(date_str) or (9999, 99, 99)
            events.append({
                "data": date_str,
                "evento": evento,
                "_sort": sort_key,
            })

        events.sort(key=lambda x: x["_sort"])
        for e in events:
            del e["_sort"]

        return events

    def extract_dispositivo(self, text: str) -> str:
        """
        Extrai a seção DISPOSITIVO de uma sentença/acórdão.
        É a seção mais determinística para predição de desfecho.
        """
        patterns = [
            r"(?:DISPOSITIVO|D\s*I\s*S\s*P\s*O\s*S\s*I\s*T\s*I\s*V\s*O)\s*\n([\s\S]{100,3000}?)(?:\n[A-Z]{4,}|\Z)",
            r"(?:Ante o exposto|Diante do exposto|Ex positis|Por tais fundamentos)[,:]?\s*([\s\S]{100,2000}?)(?:\n[A-Z]{4,}|\Z)",
            r"(?:JULGO|Julgo)\s+(?:PROCEDENTE|IMPROCEDENTE|PARCIALMENTE)([\s\S]{0,1500}?)(?:\n[A-Z]{4,}|\Z)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.MULTILINE)
            if m:
                trecho = m.group(0).strip()
                if len(trecho) >= 80:
                    return trecho[:2000]
        return ""

    def _unique_preserve_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in items:
            key = x.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(x)
        return out

    def get_representative_chunk(
        self,
        chunks: list[str],
        max_tokens: int = 512,
    ) -> str:
        if not chunks:
            return ""
        best_idx = 0
        best_score = -1.0
        for i, ch in enumerate(chunks):
            words = ch.split()
            if not words:
                continue
            unique_ratio = len(set(words)) / max(len(words), 1)
            length_factor = min(len(words), max_tokens) / max_tokens
            score = unique_ratio * 0.6 + length_factor * 0.4
            if score > best_score:
                best_score = score
                best_idx = i

        chosen = chunks[best_idx]
        words = chosen.split()
        if len(words) >= max_tokens // 2:
            if len(words) <= max_tokens:
                return chosen
            return " ".join(words[:max_tokens])

        acc: list[str] = []
        for ch in chunks:
            for w in ch.split():
                acc.append(w)
                if len(acc) >= max_tokens:
                    return " ".join(acc[:max_tokens])
        return " ".join(acc)
