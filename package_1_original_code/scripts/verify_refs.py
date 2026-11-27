"""
Kaynakça Doğrulama Betiği (Bibliographic Verification Script) v3

Vize/proposal.html içerisindeki 10 adet kaynakça maddesini programatik olarak
arXiv API ve Crossref API üzerinden sorgular.

Kural:
- Bellekten hiçbir 'MATCH' üretilmez.
- Yalnızca API'den dönen gerçek verilerle (başlık, yazarlar, yıl) kıyaslanır.
- Yazar uyumsuzluğu varsa doğrudan 'MISMATCH' olarak etiketlenir.
- proposal.html dosyasına ASLA dokunulmaz, yalnızca raporlama yapılır.
"""

import os
import sys
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
import difflib

REFERENCES = [
    {
        "id": "ref-1",
        "key": "Wang et al. (2021)",
        "claimed_title": "Tent: Fully Test-Time Adaptation by Entropy Minimization",
        "claimed_authors": ["Wang, D.", "Shelhamer, E.", "Liu, S.", "Olshausen, B.", "Darrell, T."],
        "claimed_year": 2021,
        "claimed_venue": "International Conference on Learning Representations (ICLR 2021)",
    },
    {
        "id": "ref-2",
        "key": "Schneider et al. (2020)",
        "claimed_title": "Improving robustness against common corruptions by covariate shift adaptation",
        "claimed_authors": ["Schneider, S.", "Rusak, E.", "Eck, L.", "Bringmann, O.", "Brendel, W.", "Bethge, M."],
        "claimed_year": 2020,
        "claimed_venue": "Advances in Neural Information Processing Systems (NeurIPS 2020)",
    },
    {
        "id": "ref-3",
        "key": "Hendrycks & Dietterich (2019)",
        "claimed_title": "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations",
        "claimed_authors": ["Hendrycks, D.", "Dietterich, T."],
        "claimed_year": 2019,
        "claimed_venue": "International Conference on Learning Representations (ICLR 2019)",
    },
    {
        "id": "ref-4",
        "key": "Recht et al. (2019)",
        "claimed_title": "Do CIFAR-10 Classifiers Generalize to CIFAR-10?",
        "claimed_authors": ["Recht, B.", "Roelofs, R.", "Schmidt, L.", "Shankar, V."],
        "claimed_year": 2019,
        "claimed_venue": "International Conference on Machine Learning (ICML 2019)",
    },
    {
        "id": "ref-5",
        "key": "Li et al. (2017)",
        "claimed_title": "Deeper, Broader and Artier Domain Generalization",
        "claimed_authors": ["Li, D.", "Yang, Y.", "Song, Y. Z.", "Hospedales, T. M."],
        "claimed_year": 2017,
        "claimed_venue": "IEEE International Conference on Computer Vision (ICCV 2017) [PACS]",
    },
    {
        "id": "ref-6",
        "key": "Niu et al. (2022)",
        "claimed_title": "Efficient Test-Time Model Adaptation without Forgetting",
        "claimed_authors": ["Niu, S.", "Wu, J.", "Zhang, Y.", "Chen, Y.", "Zheng, S.", "Zhao, P.", "Tan, M."],
        "claimed_year": 2022,
        "claimed_venue": "International Conference on Machine Learning (ICML 2022) [EATA]",
    },
    {
        "id": "ref-7",
        "key": "Wang et al. (2022)",
        "claimed_title": "Continual Test-Time Domain Adaptation",
        "claimed_authors": ["Wang, Q.", "Fink, O.", "Van Gool, L.", "Dai, D."],
        "claimed_year": 2022,
        "claimed_venue": "IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2022) [CoTTA]",
    },
    {
        "id": "ref-8",
        "key": "Niu et al. (2023)",
        "claimed_title": "Towards Stable Test-Time Adaptation in Dynamic Wild World",
        "claimed_authors": ["Niu, S.", "Wu, J.", "Zhang, Y.", "Wen, Z.", "Chen, Y.", "Zhao, P.", "Tan, M."],
        "claimed_year": 2023,
        "claimed_venue": "International Conference on Learning Representations (ICLR 2023) [SAR - İlgili Çalışmalar]",
    },
    {
        "id": "ref-9",
        "key": "Zhao et al. (2023)",
        "claimed_title": "On the Pitfalls of Test-Time Adaptation",
        "claimed_authors": ["Zhao, B.", "Chen, C.", "Zhang, S."],
        "claimed_year": 2023,
        "claimed_venue": "arXiv preprint arXiv:2303.17576 [İlgili Çalışmalar]",
    },
    {
        "id": "ref-10",
        "key": "Guo et al. (2017)",
        "claimed_title": "On Calibration of Modern Neural Networks",
        "claimed_authors": ["Guo, C.", "Pleiss, G.", "Sun, Y.", "Weinberger, K. Q."],
        "claimed_year": 2017,
        "claimed_venue": "International Conference on Machine Learning (ICML 2017) [ECE Metriği]",
    },
]


def clean_str(s: str) -> str:
    """Küçük harf ve alfanümerik temizlik."""
    return "".join(c.lower() for c in s if c.isalnum() or c.isspace()).strip()


def query_arxiv(title: str) -> Optional[Dict[str, Any]]:
    """arXiv API ile başlık sorgusu."""
    base_url = "http://export.arxiv.org/api/query"
    clean_t = title.replace('"', '').replace(':', ' ').strip()
    encoded = urllib.parse.quote(f'ti:"{clean_t}"')
    url = f"{base_url}?search_query={encoded}&max_results=3"
    req = urllib.request.Request(url, headers={"User-Agent": "AntigravityRefVerifier/3.0"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            xml_text = resp.read().decode("utf-8")
            root = ET.fromstring(xml_text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            if not entries:
                return None
            
            best_entry = None
            best_ratio = 0.0
            for entry in entries:
                title_elem = entry.find("atom:title", ns)
                if title_elem is not None and title_elem.text:
                    t_cand = " ".join(title_elem.text.split())
                    r = difflib.SequenceMatcher(None, clean_str(title), clean_str(t_cand)).ratio()
                    if r > best_ratio:
                        best_ratio = r
                        best_entry = entry

            if best_entry is not None and best_ratio >= 0.70:
                title_elem = best_entry.find("atom:title", ns)
                ret_title = " ".join(title_elem.text.split()) if title_elem is not None else ""
                ret_authors = [
                    a.find("atom:name", ns).text
                    for a in best_entry.findall("atom:author", ns)
                    if a.find("atom:name", ns) is not None and a.find("atom:name", ns).text
                ]
                published_elem = best_entry.find("atom:published", ns)
                ret_year = int(published_elem.text[:4]) if published_elem is not None and published_elem.text else None
                id_elem = best_entry.find("atom:id", ns)
                ret_id = id_elem.text if id_elem is not None else url

                return {
                    "source": "arXiv",
                    "url": url,
                    "paper_url": ret_id,
                    "title": ret_title,
                    "authors": ret_authors,
                    "year": ret_year,
                }
    except Exception as e:
        return {"source": "arXiv", "url": url, "error": str(e)}
    return None


def query_crossref(title: str) -> Optional[Dict[str, Any]]:
    """Crossref API ile başlık sorgusu."""
    base_url = "https://api.crossref.org/works"
    params = {
        "query.bibliographic": title,
        "rows": 3,
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    headers = {
        "User-Agent": "AntigravityRefVerifier/3.0 (mailto:ozalomerli011@gmail.com)"
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("message", {}).get("items", [])
            if not items:
                return None
            
            best_item = None
            best_ratio = 0.0
            for item in items:
                titles = item.get("title", [])
                if titles:
                    t_cand = titles[0]
                    r = difflib.SequenceMatcher(None, clean_str(title), clean_str(t_cand)).ratio()
                    if r > best_ratio:
                        best_ratio = r
                        best_item = item

            if best_item is not None and best_ratio >= 0.70:
                ret_title = best_item.get("title", [""])[0]
                ret_authors = []
                for a in best_item.get("author", []):
                    given = a.get("given", "")
                    family = a.get("family", "")
                    ret_authors.append(f"{family}, {given}".strip(", "))
                
                ret_year = None
                pub_parts = best_item.get("published", {}).get("date-parts", [[]])[0]
                if pub_parts:
                    ret_year = pub_parts[0]
                elif best_item.get("created", {}).get("date-parts", [[]])[0]:
                    ret_year = best_item.get("created", {}).get("date-parts", [[]])[0][0]

                return {
                    "source": "Crossref",
                    "url": url,
                    "title": ret_title,
                    "authors": ret_authors,
                    "year": ret_year,
                }
    except Exception as e:
        return {"source": "Crossref", "url": url, "error": str(e)}
    return None


def verify_reference(ref: Dict[str, Any]) -> Dict[str, Any]:
    """Bir referansı sorgular ve MATCH / MISMATCH / NOT FOUND / UNCHECKED belirler."""
    title = ref["claimed_title"]
    print(f"Doğrulanıyor: {ref['key']} -> '{title[:45]}...'")

    arxiv_res = query_arxiv(title)
    time.sleep(1)

    crossref_res = query_crossref(title)
    time.sleep(1)

    candidates = []
    if arxiv_res and "error" not in arxiv_res and arxiv_res.get("title"):
        candidates.append(arxiv_res)
    if crossref_res and "error" not in crossref_res and crossref_res.get("title"):
        candidates.append(crossref_res)

    if not candidates:
        errors = []
        if arxiv_res and "error" in arxiv_res:
            errors.append(f"arXiv: {arxiv_res['error']}")
        if crossref_res and "error" in crossref_res:
            errors.append(f"Crossref: {crossref_res['error']}")
        
        if errors:
            return {
                "status": "UNCHECKED",
                "details": "Ağ veya API hatası: " + " | ".join(errors),
                "url": (arxiv_res or crossref_res or {}).get("url", "N/A"),
                "returned_title": "N/A",
                "returned_authors": "N/A",
                "returned_year": "N/A",
                "source": "None",
            }
        else:
            return {
                "status": "NOT FOUND",
                "details": "Bibliyografik kayıt bulunamadı.",
                "url": (arxiv_res or crossref_res or {}).get("url", "N/A"),
                "returned_title": "N/A",
                "returned_authors": "N/A",
                "returned_year": "N/A",
                "source": "None",
            }

    best = None
    best_ratio = -1.0
    for cand in candidates:
        r = difflib.SequenceMatcher(None, clean_str(title), clean_str(cand["title"])).ratio()
        if r > best_ratio:
            best_ratio = r
            best = cand

    ret_title = best["title"]
    ret_authors = best["authors"]
    ret_year = best["year"]
    ret_url = best.get("url")

    # Başlık benzerliği
    title_ratio = difflib.SequenceMatcher(None, clean_str(title), clean_str(ret_title)).ratio()
    
    # Yıl kontrolü (+-1 tolerans)
    claimed_year = ref["claimed_year"]
    year_match = (ret_year == claimed_year) or (ret_year in [claimed_year - 1, claimed_year + 1])

    # Yazar kontrolü: İddia edilen ilk yazar soyadı dönen yazarlarda var mı?
    # Ayrıca iddia edilen ilk yazar, dönen ilk yazarla örtüşüyor mu?
    claimed_first_surname = ref["claimed_authors"][0].split(",")[0].lower().strip()
    
    ret_authors_str = " ".join(ret_authors).lower()
    first_author_found = claimed_first_surname in ret_authors_str
    
    # Dönen ilk yazar kim?
    ret_first_author = ret_authors[0].lower() if ret_authors else ""
    first_author_is_first = claimed_first_surname in ret_first_author

    # Durum kararı:
    # Başlık eşleşiyor (%85+) ve ilk yazar doğru ise MATCH.
    # Başlık eşleşse bile yazarlar tamamen farklıysa MISMATCH (ör. Zhao et al. iddia edilip Hao Zhao, Yuejiang Liu dönmesi ve diğer yazarların tutmaması).
    # Kaynak 9 özelinde: İddia: Zhao, B., Chen, C., Zhang, S. -> Dönen: Hao Zhao, Yuejiang Liu, Alexandre Alahi, Tao Lin.
    # Burada soyadı Zhao tutsa bile B. Zhao değil Hao Zhao ve ortak yazarlar tamamen uyumsuz!
    
    mismatch_reason = []
    if title_ratio < 0.85:
        mismatch_reason.append(f"Başlık benzerliği düşük (%{title_ratio*100:.1f})")
    if not first_author_found:
        mismatch_reason.append(f"İlk yazar '{claimed_first_surname}' API yazarlarında bulunamadı")
    if not year_match:
        mismatch_reason.append(f"Yıl uyuşmazlığı ({claimed_year} vs {ret_year})")

    # Tüm yazarların örtüşme denetimi (en az 2 yazar soyadı eşleşmeli veya tek yazarlı ise tek)
    matched_authors_count = 0
    for a in ref["claimed_authors"]:
        sname = a.split(",")[0].lower().strip()
        if sname in ret_authors_str:
            matched_authors_count += 1

    if len(ref["claimed_authors"]) > 1 and matched_authors_count < 2:
        mismatch_reason.append(f"Yazar kadrosu uyumsuz (Yalnızca {matched_authors_count} yazar eşleşti)")

    if not mismatch_reason:
        status = "MATCH"
    else:
        status = "MISMATCH"

    return {
        "status": status,
        "mismatch_reason": "; ".join(mismatch_reason) if mismatch_reason else "None",
        "title_ratio": title_ratio,
        "returned_title": ret_title,
        "returned_authors": ", ".join(ret_authors) if isinstance(ret_authors, list) else str(ret_authors),
        "returned_year": ret_year,
        "url": ret_url,
        "source": best.get("source"),
    }


def main():
    report_lines = [
        "# Vize/proposal.html Kaynakça Programatik Doğrulama Raporu",
        "",
        "Bu rapor, `Vize/proposal.html` içerisindeki 10 adet kaynakça maddesinin gerçek bibliyografik API'ler (arXiv API ve Crossref API) üzerinden başlık sorgulamasıyla programatik olarak doğrulanması sonucunda üretilmiştir.",
        "",
        "> [!IMPORTANT]",
        "> **Doğrulama İlkeleri:**",
        "> 1. Bellekten kesinlikle hiçbir 'MATCH' üretilmemiştir. Yalnızca API tarafından doğrudan döndürülen alanlar kıyaslanmıştır.",
        "> 2. Ağ veya API çağrısı başarısız olduğunda 'UNCHECKED' etiketi atanır.",
        "> 3. `proposal.html` içeriğine hiçbir müdahalede bulunulmamıştır (Salt Raporlama).",
        "",
        "| No | Anahtar / Yazar | İddia Edilen Başlık | API Kaynağı | Durum | Başlık Benzerliği | Tespit / Not |",
        "|---|---|---|---|:---:|:---:|---|",
    ]

    detailed_sections = []

    for idx, ref in enumerate(REFERENCES, 1):
        res = verify_reference(ref)
        status = res["status"]
        ratio_str = f"%{res.get('title_ratio', 0.0)*100:.1f}" if "title_ratio" in res else "N/A"
        source = res.get("source", "N/A")
        note = res.get("mismatch_reason", "-")
        report_lines.append(
            f"| {idx} | {ref['key']} | {ref['claimed_title'][:32]}... | {source} | **`{status}`** | {ratio_str} | {note} |"
        )

        detail_md = f"""
### Kaynak {idx}: {ref['key']}

- **Doğrulama Durumu:** `{status}`
- **Sorgulanan Servis:** {source}
- **Kullanılan Sorgu URL'si:** [{res.get('url')}]({res.get('url')})
- **Başlık Eşleşme Oranı:** {ratio_str}
- **Uyuşmazlık Notu:** {note}

| Bibliyografik Alan | İddia Edilen (proposal.html) | API Tarafından Döndürülen |
|---|---|---|
| **Başlık** | {ref['claimed_title']} | {res.get('returned_title')} |
| **Yazarlar** | {', '.join(ref['claimed_authors'])} | {res.get('returned_authors')} |
| **Yıl** | {ref['claimed_year']} | {res.get('returned_year')} |
| **Yayın Yeri / Not** | {ref['claimed_venue']} | - |
"""
        detailed_sections.append(detail_md)

    full_content = "\n".join(report_lines) + "\n\n---\n\n## Detaylı Karşılaştırmalar\n" + "\n".join(detailed_sections)

    out_path = os.path.join("Vize", "kaynakca_dogrulama.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"\nRapor başarıyla güncellendi -> {out_path}")


if __name__ == "__main__":
    main()
