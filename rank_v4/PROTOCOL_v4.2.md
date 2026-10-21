# Sıralama deneyi protokolü v4.2 (+ teknik ayrıntılar, 2026-10-21)

Durum: **pilot öncesi taslak**. Doğrulayıcı test koşusu bu belgenin "test öncesi kayıt" bölümü doldurulup onaylanmadan başlatılmaz.
Önceki sürümler (v1–v4.2) sohbet kaydındadır; bu dosya bunları tek metinde toplar ve 2026-10-21 teknik ayrıntılarını ekler.

## 1. Kapsam ve iddia
- Senaryo S1: **önceden işaretlenmiş üründe yorum moderasyonu**. Her koşuda tek hedef ürün saldırıya uğrar; silme yalnızca hedef ürünün yorumlarında yapılır. Hedefin bilinmesi oracle varsayımıdır (sınırlılık). Uçtan uca saldırı tespiti iddia edilmez.
- İnferans: sabit Trendyol test kohortuna ve tanımlı saldırı üreticisine koşullu. Ürün evrenine genelleme iddia edilmez.
- S2 (küresel bütçe), A4, A5, ürün bootstrap'ı: ikincil/betimsel.

## 2. Veri, bölmeler, kaynak havuzu
- Sıralama havuzu: değerlendirme yarısında ≥40 yorumlu ürünler (`ranking.py` satır 37 ile aynı tanım).
- Bölme: havuzun 192-ürün temiz Bayes sıralamasındaki yarılara göre tabakalı, 1/3 doğrulama – 2/3 test (`SeedSequence(ENTROPY, spawn_key=(0,))`). Her bölme kendi sıralama evrenidir; hedef yarıları bölmenin kendi temiz sıralamasına göre tanımlanır.
- Kaynak ürünler: referans yarısındaki ürünlerin %30'u (`spawn_key=(1,)`) ridge eğitiminden çıkarılır; ridge (α=10) kalan referans ürünlerinde yeniden eğitilir ve yalnızca bu deneyde kullanılır.
- Uygun havuz P_{f,p}: kaynak ürünlerde, kategori = cat(p), yıldız ∈ S_f (A1/A3: {1,2}; A2: {5}), metni (strip + NFC + casefold) hedefteki hiçbir metinle eşleşmeyen yorumlar. Koşul |P_{f,p}| ≥ k_p. Sağlanmazsa hedef o aileden dışlanır (A2'den dışlanan A3'ten de). α değeri, dışlama oranı her iki bölmede ≤%10 ise uygulanabilir.

## 3. Saldırı aileleri
- A1 tutarsız şişirme: olumsuz metin (kaynak 1–2★), 5★, alt yarı. Birincil.
- A2 tutarsız düşürme: olumlu metin (kaynak 5★), 1★, üst yarı. Birincil.
- A3 tutarlı bombalama: olumsuz metin (kaynak 1–2★), 1★, üst yarı; A2 ile aynı hedef/α/ρ/r. Negatif kontrol, keşifsel.
- k_p = ⌈α_f·n_p⌉, α_f ∈ {0,10; 0,20; 0,30}; seçim doğrulamada: normalize filtresiz kayma medyan D/(|U|−1) ≥ 0,08 sağlayan en küçük uygulanabilir α.

## 4. Bütçe
- b = ⌈ρ·k_p⌉, üst sınır b ≤ n_p + k_p − 10 (kırpma sayılır). ρ ∈ {0,5; 1; 2; 4}; birincil ρ = 1.
- Ana eğri: b/(n_p+k_p) ≤ %25 olan hücreler; diğerleri stres testi (keşifsel).
- FF kontrolünde b_c = ⌈ρ·⌈α_f·n_c⌉⌉, aynı üst sınırla (k = 0).

## 5. Yöntemler
B0 (silme yok); B2' (hedef içi rastgele); R2 (önce `rating_i == 1` olanlar, sonra kalanlar; her iki grup içinde π_rand); R3 (`rating_i` artan, sonra π_rand); R4 (|rating_i − hedefin diğer yorumlarının ortalaması|); SAAD adayları: rating-model gap, ensemble gap, SAAD-core, sınıf içi kalibre gap (karar bekliyor); LAF (etiket-bilen filtre, yalnızca ρ = 0,5).

## 6. Koşu içi hesaplama sırası
0. Bir kez: evren = bölmedeki ürünlerin orijinal yorumları; μ = bunların ortalama puanı (sabit); diğer ürünlerin temiz Bayes puanları (m = 20); temiz context distance; pretrained sentiment ve ridge tahmini yorum başına.
1. Saldırı: kaynak çekimi `spawn_key=(2, aile, bölme, r, p)`, hedef içinde yerine koymadan; enjekte yorumlar hedefe eklenir.
2. Skorlar: robust z medyan/MAD saldırılı evrenin tüm yorumları üzerinden; context distance yalnızca hedefte yeniden; R4 hedefin bırak-bir-dışarı ortalamasıyla.
3. **Sonluluk kontrolü:** tüm skorlar `numpy.isfinite`; NaN veya ±∞ varsa koşu açık hata ile durur (sessiz sıralama yok).
4. Seçim: hedefin yorumları (−s, π_tie, rid) sözlük sırasıyla; s ham float64 skordur, tolerans yok; ilk b yorum silinir.
5. Puan: (Σr + 20μ)/(n + 20) kalan yorumlarla.
6. Sıra: rank(x) = 1 + #{q≠p: s_q > x} + ½·#{q≠p: s_q = x} (tam eşitlik); D = |rank(s^filtreli) − rank(s^temiz)|.

## 7. Kimlikler, hash önceliği ve tohumlar
- **Enjekte yorum rid'i (kesin biçim):** `inj|v1|{aşama}|{aile}|{bölme}|{r}|{pid}|{j}|{kaynak_rid}`
  - aşama ∈ {`attack`}, aile ∈ {`A1`,`A2`,`A3`}, bölme ∈ {`val`,`test`}, r ve pid ondalık ASCII (başında sıfır yok), j = hedef içindeki 0 tabanlı çekim sırası (ondalık ASCII), kaynak_rid = kaynak yorumun corpus rid'i (`tr-<int>`).
  - Ayraç `|`; hiçbir bileşen `|` içeremez (kodda assert). Kodlama ASCII (UTF-8 alt kümesi).
- **Hash önceliği:** π = int.from_bytes(blake2b(msg, digest_size=8, key=ENTROPY.to_bytes(8,"big")).digest(), "big").
  msg = 10 alanın her biri için [4 bayt big-endian uzunluk][UTF-8 bayt]: (1) `v1`, (2) namespace `tie`|`rand`, (3) aşama `attack`|`ff`, (4) aile, (5) bölme, (6) tekrar (FF'de `-`), (7) birim türü `target`|`control`, (8) birim pid, (9) yöntem kimliği (`tie` için `*`; `rand` için yöntem adı), (10) rid (NFC).
  - Rastgele baseline anahtarları tüm senaryo alanlarını içerir: aşama, aile, bölme, tekrar, hedef/kontrol, yöntem kimliği, yorum kimliği.
  - Hash eşitliğinde rid'in Unicode kod noktası sırası son çözümdür.
- **SeedSequence anahtarları:** (0,) bölme; (1,) kaynak ürünler; (2, aile_no, bölme_no, r, pid) kaynak çekimi; (5, pid) FF eşleştirme eşitliği. aile_no A1=1, A2=2, A3=3; bölme_no val=0, test=1. Farklı anahtarlar için akış çakışması imkânsız değildir, olasılığı ihmal edilebilir; kullanılan anahtarların benzersizliği birim testle doğrulanır.
- **Sınır yakınlığı tanısı (kararı etkilemez):** sınırdaki iki skor s_b ve s_{b+1} için u = max(math.ulp(s_b), math.ulp(s_{b+1})); |s_b − s_{b+1}| ≤ 4u ise koşu "ulp-yakın sınır" olarak sayılır.
- Yeniden üretilebilirlik: aynı veri hash'leri, ENTROPY, kod sürümü ve ortam (Python/NumPy/sklearn sürümleri, donanım) ile aynı çıktı.

## 8. Metrikler
Hedef: D (birincil), puan sapması, yakalama oranı, TRR_p = korunan orijinal 1–2★ / L_p (L_p = 0 → tanımsız, K analizinden çıkar), yıldız sınıfı silme oranları, b/L_p katmanı. FF: aynı ölçütler kontrol üründe. A3: hedefin 1★ havuzunda ayrım AUC'si, zenginleşme, YO_A2 − YO_A3.

## 9. Tekrar, MCSE, aralıklar
- Her r tam bağımsız tekrar; kaynaklar hedef içinde yerine koymadan, hedefler ve tekrarlar arası yeniden kullanım serbest.
- θ_K,f = ortalama_r[TRR_r(S*) − TRR_r(R*_f)]; MCSE = SD_r/√R; aralık θ̂ ± t_{R−1;0,975}·MCSE.
- θ_D,rel,f = Σ_r N_r / Σ_r M_r (N_r = Σ_p[D(S*) − D(R*_f)], M_r = Σ_p D(B0)); MCSE = delete-one-replicate jackknife SE; aralık θ̂ ± t_{R−1;0,975}·SE_J (jackknife SE ile kurulan t aralığı). CV(M_r) > 0,20 ise tekrar bootstrap'ı duyarlılık olarak raporlanır.
- Eşikler: MCSE(θ_D,rel) ≤ 0,025; MCSE(θ_K) ≤ 0,0125. R₀ = 10, +5, R_max = 30 (doğrulamada projeksiyonla); testte gerçekleşen MCSE eşiği aşarsa nicelik keşifsel.
- FF'de skor tabanlı yöntemler deterministik, bir kez; rastgele baseline'lar R üzerinden.

## 10. Hipotez ve karar
- Aile bazında (A1, A2) H1: θ_D,rel,f aralığının üst sınırı < 0,10 (non-inferiority) VE θ_K,f aralığının alt sınırı > 0 (superiority; yorum eşiği 5 yp). Aile içinde IUT, aileler arası Holm (α = 0,05).
- Doğrulayıcı sayılma koşulları: her ailede dışlamalardan sonra L_p ≥ 1 olan ≥30 test hedefi; MCSE eşikleri; manipülasyon kontrolü eşiği (karar bekliyor; öneri ≥%80 beklenen polarite); FF eşleştirme SMD < 0,1 (aksi halde FF betimsel).

## 11. Açık kararlar (pilot için geçici değerler kullanıldı)
Kaynak ürün oranı (%30 geçici), kalibre gap'in adaylığı (pilota dahil), manipülasyon kontrolü eşiği, ENTROPY değeri (pilotta 20261021).

## 12. Test öncesi kayıt (doğrulama sonrası doldurulacak)
α_f, S*, R*_f, R, uygunluk sayıları, FF eşleşmeleri, dışlanan hedefler, sınıflama (doğrulayıcı/keşifsel), ortam ve hash'ler.

## 13. Pilot notları (2026-10-21) — onay bekleyen açıklamalar
- **LAF uygulama hatası düzeltildi (protokole uygunluk):** ilk kod LAF'ta bütçenin kalanını orijinal yorumlarla dolduruyordu; protokol (v3–v4) LAF'ın yalnızca min(b, k_p) enjekte yorumu sildiğini söylüyor. Kod protokole getirildi; protokol değişmedi.
- **Sınır tanısı ikiye ayrıldı (öneri):** `exact` (sınırdaki iki skor tam eşit; π ile çözülür; ayrık puanlı yöntemlerde beklenen durum) ve `near` (0 < |fark| ≤ 4·max ulp). Silme kuralını etkilemez.
- **Kaynak havuzu normalize metne göre tekilleştirildi (açıklama):** iki kaynak yorumun normalize metni aynıysa yalnızca en küçük corpus kimlikli olan havuzda kalır; böylece hedefe aynı metin iki kez eklenemez.
- **Bölme tanımı (açıklama):** tabakalama 192-ürün temiz sıralamasının yarılarına göre; hedef yarıları her bölmenin kendi temiz sıralamasına göre (val 32/32, test 64/64).

## 14. Önceden dondurulmuş kararlar (2026-10-21, doğrulama seçim koşusundan ve testten önce)
Bu bölüm 1–13'teki çelişen ifadelerin yerine geçer. Kod karşılıkları `code/common.py` "frozen decision rules" bölümünde; metadata `out/frozen_metadata.json` dosyasında.

**14.1 Statüler.** A1 doğrulayıcı aday (yalnızca α ∈ {0,10; 0,20}). A2 keşifsel (yalnızca onarım hipotezi eklenmez). A3 betimsel. FF, bütçe eğrisi, stres testleri, S2 ve duyarlılık analizleri keşifsel.

**14.2 Sıra ve seçim kuralları (yalnızca A1 doğrulama verisi, ρ = 1).**
1. α_A1: {0,10; 0,20} içinde, dışlanan hedef oranı her iki bölmede ≤ %10 ve doğrulamada B0'ın medyan normalize kayması D/(|U|−1) ≥ 0,08 olan en küçük değer. Hiçbiri sağlamazsa sıralama deneyinin tamamı keşifsel (`select_alpha`).
2. R*_A1: {R2, R3, R4} içinde en küçük ham ortalama D; eşitlikte R2, R3, R4 sırası (`select_r_star`).
3. S*: adaylar gap_ens, saad_core, gap_rm, gap_cal (kalibre gap dahil). θ_D,rel ve θ_K, R*_A1'e göre hesaplanır. Önce θ_D,rel nokta tahmini ≤ 0,10 olanlar; bunlar arasında en yüksek θ_K. Böyle aday yoksa en küçük θ_D,rel. Tam eşitlikte gap_ens, saad_core, gap_rm, gap_cal sırası. Sonlu olmayan istatistikli aday seçilmez (`select_s_star`).
4. R: R₀ = 10, +5, R_max = 30; MCSE eşikleri yalnızca A1 için (θ_D,rel ≤ 0,025; θ_K ≤ 0,0125), doğrulamadan teste projeksiyonla.
5. **Tanımlar (2026-10-21, kullanıcı tarafından donduruldu):**
   - *α medyanı:* havuz koşulunu sağlayan her A1 doğrulama hedefi için (L_p = 0 olanlar dahil) ilk R₀ = 10 tekrardaki B0 hedef sıra kaymalarının ortalaması; bu hedef ortalamalarının medyanı; |U| − 1'e bölünür (`median_norm_target_mean`).
   - *MCSE projeksiyonu (outcome-specific):* MCSE_test ≈ MCSE_val·√(n_val/n_test); D için havuz koşulunu sağlayan A1 hedefleri (doğrulama 32, test 64), K için L_p ≥ 1 olan A1 hedefleri (doğrulama 30, test 56). Testten yalnızca bu saldırı öncesi sayımlar kullanılır; kodda sayımlar bu değerlerle doğrulanır, farklıysa koşu durur.
   - *Tekrarlar:* α_A1, R*_A1 ve S* ilk 10 doğrulama tekrarıyla (r = 0…9) seçilir ve dondurulur. R artırılırsa α ve yöntemler yeniden seçilmez; ek tekrarlar (r = 10, 11, …) yalnızca MCSE/R kuralı için kullanılır. MCSE, S* ile R*_A1 karşıtlığı üzerinden hesaplanır.

**14.3 Uygun hedef sayımı.** Bir hedef, kaynak havuzu ≥ k_p ve L_p ≥ 1 ise uygundur (A3'te ayrıca eşli A2 havuzu). FF eşleşmesi, bütçe kırpması ve sınır tanısı uygunluğu etkilemez. Koşul: her iki bölmede ≥ 30 (eşik dahil) (`eligible_targets`, `exp.eligibility`).

**14.4 Hipotez ve karar kuralı (A1).** H1_A1 doğrulanmış sayılır ancak ve ancak (i) θ_D,rel için %95 aralığın üst sınırı < 0,10 VE (ii) θ_K için %95 aralığın alt sınırı > 0. Her sınır tek yönlü 0,025 düzeyinde bir testtir; tek doğrulayıcı aile olduğundan çoklu karşılaştırma düzeltmesi yoktur (`h1_decision`). Aralıklar: θ_K için θ̂ ± t_{R−1;0,975}·SD_r/√R; θ_D,rel için delete-one-replicate jackknife SE ile kurulan t aralığı. A1 şu durumlardan herhangi birinde keşifsel sınıflanır: α kuralı başarısız; bir bölmede < 30 uygun hedef; polarite kontrolü geçilmedi veya yapılmadı; testte gerçekleşen MCSE eşiği aştı (`classify_a1`).

**14.5 K ölçütü.** Birincil: toplamların oranı TRR farkı (`theta_k_replicate`). Tanı: (ΣL_p)²/ΣL_p², yalnızca ağırlık yoğunlaşması göstergesi; bağımsız veya etkin örneklem büyüklüğü olarak yorumlanmaz (`weight_concentration`). Keşifsel duyarlılık: hedeflere eşit ağırlıklı ortalama TRR_p farkı (`theta_k_unweighted_replicate`).

**14.6 Kaynak ürünler ve ridge.** Mevcut %30 seçim (SeedSequence(20261021, (1,))) korunur; liste ve hash `frozen_metadata.json`'da. Kaynak seçimi sonuçlara göre değiştirilmez. Sıralama deneyi yalnızca ridge_srcout kullanır; makale sonuçları ve eski deney denetimi ridge_full'dur. Her sonuç dosyası model etiketi, eğitim satırı sayısı ve model hash'ini içerir (`model_meta`); iki modelin sayıları aynı tabloda birleştirilmez.

**14.7 Kaynak polarite kontrolü (insan).**
- Havuzlar: A1 = kaynak ürünlerdeki tüm 1–2★ yorumlar (A3 de bu havuzu kullanır); A2 = kaynak ürünlerdeki tüm 5★ yorumlar (`exp.polarity_pool_rids`).
- Örneklem (2026-10-21 değişikliği, kullanıcı onaylı): aile başına 100 metin, kategorilere planlanan saldırı çekimleriyle orantılı kotalarla dağıtılır. Ağırlık w_c = ailenin hedef ürünlerinde (doğrulama + test) kategori c'deki Σ n_p (k_p = ⌈α·n_p⌉ her α için n_p ile orantılı); kotalar en büyük kalan yöntemiyle 100'e tamamlanır (`largest_remainder`). Her kategoride kaynak yorumlar corpus-id sırasına konur ve SeedSequence(ENTROPY, (9, aile_no, kategori_sırası)) permütasyonuyla gezilir; normalize metni (strip + NFC + casefold) daha önce kabul edilmiş bir metinle (aynı aile veya diğer aile; önce A1, sonra A2 işlenir) aynı olan atlanır ve sıradaki çekilir (`stratified_unique_sample`). Kategori havuzu biterse eksik başka kategoriye aktarılmaz, raporlanır. A3 bu kontrolle temsil edilmez (betimsel). Önceki `polarity_sample` (tabakasız) kullanılmaz.
- Etiketleme körlemesine: iki ailenin 200 metni SeedSequence(ENTROPY, (9, 0)) ile karıştırılır; anotatör yıldızı ve aileyi görmez; her metne "olumlu", "olumsuz" veya "belirsiz" der.
- Hesaplama: `polarity_map` ile beklenen (A1: olumsuz, A2: olumlu) / beklenmeyen / belirsiz; pay = beklenen / 100 (belirsiz beklenmeyen sayılır); geçme koşulu pay ≥ 0,80 (`polarity_result`). A1 ve A2 ayrı değerlendirilir. Geçilmezse ilgili aile keşifsel; A1 geçmezse sıralama deneyinin tamamı keşifsel.
- Bu kontrol tamamlanmadan doğrulama seçim koşusu başlatılmaz.

**14.8 Sınır tanısı.** `exact` ve `near` ayrı raporlanır; silme kararını etkilemez.

**14.9 Veri tanımı.** "Veri kartı (README, satır 28 ve 43) 500 ürün (kategori başına 100) belirtmektedir. Yayımlanan dosya `trendyol_reviews_20251212_000835.csv` 582.601 yorum ve 476 ayrı ürün sayfası (kategori başına 89–99) içermektedir; bu çalışmada 476 ürünün tamamı kullanılmıştır." Eksik ürünlerin nedeni belirtilmez. Hugging Face deposu `Doukan/trendyol-turkish-product-reviews`, revision `3b390598dac913449a75b624a06e985a1c5d726c`.

**14.10 Sabitler.** ENTROPY = 20261021; kaynak oranı %30; ridge α = 10; m = 20; MIN_KEEP = 10; ρ ∈ {0,5; 1; 2; 4} (birincil 1; ana eğri b/(n+k) ≤ %25); marj 0,10; K yorum eşiği 5 yp; uygunluk ≥ 30; polarite ≥ 0,80, n = 100/aile.

## 15. Değişiklik kaydı
| Tarih | Madde | Değişiklik | Gerekçe |
|---|---|---|---|
| 2026-10-21 | 13 | LAF kodu protokole getirildi (dolgu yok) | Pilot uygulama hatası buldu |
| 2026-10-21 | 7, 14.8 | Sınır tanısı exact/near | Ayrık skorlu yöntemlerde tam eşitlik beklenen durum |
| 2026-10-21 | 10 → 14.1 | A2 H1 keşifsel; A3 betimsel; yalnızca onarım hipotezi eklenmedi | Testte L_p ≥ 1 olan 9 hedef; onarım tek başına başarı değil |
| 2026-10-21 | 3 → 14.2 | α_A1 ızgarası {0,10; 0,20}; kural başarısızsa tümü keşifsel | α = 0,30 havuz nedeniyle uygulanamıyor |
| 2026-10-21 | 14.2 | S* seçimi yalnızca A1'de, marj içi en yüksek θ_K | Aileler arası ham D ortalaması A2 ölçeğine baskın; yalnızca D'ye dayalı seçim H1'in K koşulunu yok sayıyordu |
| 2026-10-21 | 14.2 | R* eşitlik sırası R2, R3, R4 | Daha önce tanımsızdı |
| 2026-10-21 | 10 → 14.4 | Holm kaldırıldı; aralık sınırlarıyla karar kuralı | Tek doğrulayıcı aile; %95 iki yönlü aralık ile α tutarlılığı |
| 2026-10-21 | 14.3 | Uygunluk her iki bölmede ≥ 30; FF/kırpma/tanı etkilemez | Doğrulamada A1 tam eşikte |
| 2026-10-21 | 14.5 | Ağırlık yoğunlaşması tanısı ve ağırlıksız duyarlılık | Toplamların oranı birkaç büyük L_p'li hedefe baskın olabilir |
| 2026-10-21 | 14.6 | Kaynak seçimi korundu; tabakalı seçim dondurulmadı | Ridge davranışı görüldükten sonra kural değiştirmek araştırmacı serbestliği olurdu |
| 2026-10-21 | 14.7 | Polarite kontrolü kuralı, körleme ve sonuçları | Kaynak yıldızı polaritenin gürültülü göstergesi |
| 2026-10-21 | 14.9 | Veri tanımı ve HF revision | README ile CSV ürün sayısı farklı |
| 2026-10-21 | 14.7 | Polarite örneklemi kategoriye göre kotalı ve normalize metne göre aileler arası tekil | Kullanıcı talebi: örneklem gerçek kaynak çekimlerini temsil etmeli |
| 2026-10-21 | 14.2.5 | α medyanının birimi, MCSE projeksiyon sayıları ve seçim tekrarları tanımlandı | Kod incelemesi protokol metninde bu üç noktanın tanımsız olduğunu gösterdi; kullanıcı dondurdu |
| 2026-10-21 | 16 | Keşifsel analiz planı eklendi; deney keşifsel | α kuralı doğrulamada başarısız (selection_record_A1.json) |
| 2026-10-21 | 11 | Açık kararlar 14 ile kapatıldı (ENTROPY, kaynak oranı, kalibre gap, polarite eşiği) | — |

## 16. Keşifsel analiz planı (2026-10-21, test analizinden önce donduruldu)
**Statü:** α kuralı doğrulamada başarısız oldu (`out/selection_record_A1.json`); sıralama deneyinin tamamı **keşifseldir**. Hiçbir sonuç doğrulayıcı kanıt, istatistiksel üstünlük veya genelleme olarak sunulmaz. Eşik ve seçim kuralları değiştirilmedi.

1. **Aileler ve α:** A1 {0,10; 0,20}; A2 {0,10; 0,20; 0,30} (havuz açısından uygulanabilir tüm değerler); A3 {0,10; 0,20}. α seçilmez; her değer ayrı raporlanır. Havuz dışlamaları 14.3/2. bölüm kurallarıyla (A3, A2 ile eşli).
2. **Bütçe:** ana eğri ρ ∈ {0,5; 1; 2}; ρ = 4 ayrı stres testi. Protokol §4'teki %25 kuralına göre b/(n_p+k_p) > 0,25 olan ana eğri hücreleri (örn. α = 0,20, ρ = 2; α = 0,30, ρ = 1 ve 2) ayrıca "§4 stres hücresi" diye işaretlenir; hesap değişmez.
3. **Tekrar:** R = 10 sabit (test bölmesi, r = 0…9); R artırılmaz.
4. **Yöntemler:** B0, B2', R2, R3, R4, gap_rm, gap_ens, saad_core, gap_cal; LAF yalnızca ρ = 0,5. S* / R* seçilmez; kazanan ilan edilmez.
5. **Ölçütler (aile × α × ρ × yöntem):** D (hedef sıra kayması, ortalama); D_rel,m = Σ_r Σ_p D_m / Σ_r Σ_p D_B0 (saldırı kaymasının filtre sonrası kalan oranı; ikili karşılaştırma değildir); puan sapması; yakalama oranı; TRR (L_p ≥ 1 hedeflerde toplamların oranı); ağırlıksız TRR (duyarlılık); yıldız sınıfı silme oranları; ağırlık yoğunlaşması (ΣL)²/ΣL² (yalnızca tanı); sınır tanısı (exact/near); kırpma sayısı.
   A2/A3: hedefin 1★ havuzunda AUC (skor yöntemleri; yalnızca L¹ ≥ 1 hedefler), zenginleşme = Σ yakalanan / Σ beklenen rastgele-1★ yakalaması, mekanizma karşıtlığı = yakalama(A2) − yakalama(A3) (aynı r, p, α, ρ). A3'te yakalama tek başına başarı sayılmaz.
   FF: A1 ve A2 hedeflerinin test eşleşmeleri (A3 aynı kontrolleri ve bütçe kuralını paylaşır); b_c = ⌈ρ⌈α n_c⌉⌉, n_c − 10 ile sınırlı; kontroller eşleştikleri hedef sayısıyla ağırlıklı; D, puan sapması, TRR_c, 1★ silme oranı.
6. **Belirsizlik:** tekrar düzeyinde ortalama ± t_{9;0,975}·SD/√10; D_rel için delete-one-replicate jackknife SE ile t aralığı. Yalnızca sabit kohort ve saldırı üreticisi altındaki Monte Carlo belirsizliği; doğrulayıcı anlamlılık veya genelleme değil. Çoklu karşılaştırmadan "istatistiksel olarak üstün" sonucu çıkarılmaz. Ürün bootstrap'ı, S2, A4, A5 bu planda yoktur.
7. **Kayıt:** plan, parametreler, kod/veri hash'leri `out/exploratory_plan_record.json`'a yazılır, sha256'sı ayrı dosyada; analiz betiği (`explore_test.py`) çalışmadan önce bu hash'leri doğrular. Analiz sonrası kayıt değiştirilmez.
