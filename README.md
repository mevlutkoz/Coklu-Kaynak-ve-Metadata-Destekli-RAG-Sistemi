# Multi-Source RAG Pipeline — Field-Level Resolution

Farklı formatlardaki şirket belgelerinden (TXT sözleşme, CSV fiyat tablosu, JSON güncelleme logları) bilgi çeken, çelişki durumunda deterministik kurallarla en güncel kaynağı seçen bir RAG (Retrieval-Augmented Generation) pipeline'ı.

## Klasör Yapısı

```
Coklu-Kaynak-ve-Metadata-Destekli-RAG-Sistemi/
├── data/
│   ├── sozlesme.txt            # Hukuki sözleşme (9 madde)
│   ├── paket_fiyatlari.csv     # Paket bilgileri tablosu
│   └── guncellemeler.json      # Tarihli değişiklik logları
├── src/
│   ├── __init__.py
│   ├── fields.py               # Canonical field sabitleri
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── txt_parser.py       # Madde-bazlı chunking + embedding
│   │   ├── csv_parser.py       # Field-level CSV fact çıkarımı
│   │   └── json_parser.py      # Regex ile alan-bazlı JSON normalize
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── semantic_retriever.py   # Hybrid (semantic + keyword) TXT arama
│   │   └── structured_retriever.py # CSV + JSON + TXT fact toplama
│   ├── query_planner.py        # LLM intent extraction (JSON schema)
│   ├── conflict_resolver.py    # Deterministik field-level çözümleme
│   ├── llm_client.py           # Cevap üretimi (sadece resolved facts)
│   └── pipeline.py             # Orchestration
├── tests/
│   ├── __init__.py
│   └── test_rag.py             # 8 zorunlu + 7 edge-case test
├── vectorstore/                # ChromaDB persistent storage
├── main.py                     # CLI giriş noktası
├── requirements.txt
└── .env                        # OPENAI_API_KEY
```

## Kurulum

### 1. Gereksinimler

- Python 3.11+
- OpenAI API anahtarı (text-embedding-3-small ve gpt-4o erişimi)

### 2. Bağımlılıkları Kur

```bash
cd Coklu-Kaynak-ve-Metadata-Destekli-RAG-Sistemi
pip install -r requirements.txt
```

### 3. API Anahtarını Ayarla

`.env` dosyasını düzenle yoksa oluştur

```
touch .env
OPENAI_API_KEY=sk-proj-...
```

### 4. Sistemi Başlat

```bash
python main.py
```

İlk çalıştırmada `sozlesme.txt` otomatik olarak madde-bazlı chunk'lara ayrılıp ChromaDB'ye embed edilir. Sonraki çalıştırmalarda dosya değişmediyse hash kontrolü ile bu adım atlanır.

### 5. Testleri Çalıştır

```bash
# Tüm testler (test 8 için OpenAI key gerekli)
python -m pytest tests/test_rag.py -v

# Sadece offline testler (API key gereksiz)
python -m pytest tests/test_rag.py -v -k "not planner"
```

## Mimari Kararlar

### Tablo Verisi Neden Embed Edilmez

CSV tablosal veriyi klasik text-chunking ile embed etmek satır/sütun bağlamını bozar. Bir embedding "299" sayısının hangi pakete, hangi kolona ait olduğunu koruyamaz. Bu yüzden CSV hiçbir zaman vektörize edilmez; her sorguda `pd.read_csv()` ile taze okunur ve field-level fact'lere dönüştürülür:

```
paket_fiyatlari.csv → get_field_facts("Pro", ["fiyat_tl", "iade_suresi_gun"])
→ [
    {field_name: "fiyat_tl", value: 299, source_type: "csv", ...},
    {field_name: "iade_suresi_gun", value: 14, source_type: "csv", ...}
  ]
```

Aynı mantıkla JSON güncelleme logları da embed edilmez, her sorguda `json.load()` ile taze parse edilir. Böylece veri dosyasına eklenen yeni bir kayıt, restart gerektirmeden anında yansır.

### Sözleşme Metni (TXT) Nasıl Vektörize Edilir

Sözleşme yapılandırılmamış doğal dildir; semantic search burada değerlidir. `sozlesme.txt` madde-bazlı chunk'lara ayrılır (`Madde X.Y:` regex pattern) ve her chunk text-embedding-3-small ile embed edilerek ChromaDB'ye yazılır. Retrieval sırasında hybrid scoring uygulanır: cosine similarity + keyword overlap.

Hash-based reindex guard sayesinde dosya değişmediyse re-embed atlanır; dosya değiştiğinde otomatik yeniden index'lenir.

### Field-Level Conflict Resolution

Aynı bilgi (örn. iade süresi) birden fazla kaynakta farklı olabilir. Çözüm deterministik ve LLM-free'dir:

```
Öncelik:  JSON (en güncel tarihli kayıt) > CSV > TXT
```

Sistem her alan için (`fiyat_tl`, `iade_suresi_gun`, `depolama_gb`, ...) tüm kaynaklardan gelen adayları toplar, sonra Python kurallarıyla tek bir kazanan seçer. JSON'daki `extract_field_from_update()` fonksiyonu Türkçe değişiklik cümlelerini regex ile normalize eder:

```
"Pro paket iade süresi 30 güne çıkarıldı." → (iade_suresi_gun, 30)
"Pro paket fiyatı 249 TL'den 299 TL'ye güncellendi." → (fiyat_tl, 299)
```

Her resolved fact şu metadata'yı taşır:

| Alan | Açıklama |
|------|----------|
| `field_name` | Canonical alan adı |
| `value` | Seçilen değer |
| `chosen_source` | Kazanan kaynak tipi (json/csv/txt) |
| `source_file` | Kaynak dosya adı |
| `effective_date` | JSON kaydının tarihi |
| `overridden` | Başka kaynak(lar) override edildi mi |
| `overridden_source` | Override edilen dosya(lar) |

### Query Planning

LLM (gpt-4o-mini) yalnızca intent extraction yapar, karar vermez. JSON schema ile structured output kullanır:

```json
{
  "paket": "Pro",
  "asked_fields": ["fiyat_tl", "iade_suresi_gun"],
  "asks_contract_policy": true,
  "asks_current_package_info": false
}
```

Bu plan hangi kaynakların sorgulanacağını belirler: `asked_fields` varsa CSV + JSON, `asks_contract_policy` varsa TXT semantic search. Paket adı geçti diye gereksiz yere tüm kaynaklar sorgulanmaz.

### Answer Generation

LLM'e (gpt-4o) ham retrieval sonuçları verilmez; yalnızca conflict resolver'ın çıktısı (`resolved_facts`) ve ilgili sözleşme maddeleri (`policy_clauses`) verilir. LLM hangi bilginin doğru olduğuna karar vermez, sadece verilen fact'leri doğal dile çevirir.

## Veri Akışı

```
Kullanıcı Sorusu
       │
       ▼
┌──────────────┐
│ Query Planner│  gpt-4o-mini → structured JSON intent
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────┐
│           Retrieval (ihtiyaca göre)      │
│                                          │
│  CSV → field facts    (her sorguda taze) │
│  JSON → field facts   (her sorguda taze) │
│  TXT → policy clauses (semantic search)  │
│  TXT → default facts  (Madde 4.1 → 14g) │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────┐
│  Conflict Resolver   │  JSON > CSV > TXT (deterministik, LLM yok)
│  field_name bazlı    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Answer Generator   │  gpt-4o → doğal dil cevap + kaynaklar
│  (sadece resolved    │
│   facts kullanır)    │
└──────────────────────┘
```

## Örnek Sorular

| Soru | Beklenen Davranış |
|------|-------------------|
| "Pro paket fiyatı nedir ve iade süresi kaç gün?" | fiyat=299 (JSON override), iade=30 gün (JSON override) |
| "Basic paketi iptal edersem paramı geri alabilir miyim?" | iade=7 gün (JSON override) + iptal politikası (TXT Madde 5.1/5.2) |
| "Enterprise paketinde kaç GB depolama var ve destek türü nedir?" | depolama=1000 GB (JSON override, 500→1000), destek=premium destek (JSON override) |
| "Pro paketinde kaç kullanıcı limiti var?" | kullanici_limiti=30 (JSON override, en güncel 2025-03-01 kaydı 25→30) |

## Dinamik Veri Testi

Sistem restart gerektirmeden veri değişikliklerini yansıtır:

```bash
# guncellemeler.json'a yeni kayıt ekle
# Aynı soruyu tekrar sor → yeni değer anında gelir
# CSV fiyatını değiştir → bir sonraki soruda yansır
```

Bu davranış CSV ve JSON'un her sorguda taze okunması sayesinde sağlanır.
