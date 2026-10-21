# Ek not: selection_record_A1.json (sha256 571134d090222cc7699467a2834c267a24d0a26b43cedb5db27f3e93b33a51c5)

Kayıt donduruldu (salt-okunur) ve değiştirilmedi. Bu not yalnızca bir raporlama artefaktını açıklar.

- `classification` gerekçelerindeki "validation eligible 0 < 30" ve "test eligible 0 < 30" satırları gerçek uygunluk sayıları değildir.
  α kuralı başarısız olduğunda sürücü `classify_a1(None, 0, 0, ...)` çağırıyor (select_validation.py, `if alpha is None:` dalı); 0 değerleri bu çağrıdan geliyor.
- Gerçek saldırı öncesi uygunluk sayıları aynı kaydın `steps.alpha.per_alpha` alanındadır: α = 0,10 ve 0,20 için doğrulama 30, test 56 (havuz uygun 32 / 64).
- A1'in keşifsel sınıflanmasının tek geçerli nedeni: "alpha rule failed" (medyan normalize B0 kayması α = 0,10'da 0,0317, α = 0,20'de 0,0476; eşik 0,08).

## Ek 2 (2026-10-21, append-only; özgün kayıt ve yukarıdaki metin değiştirilmedi)
- `selection_record_A1.json` içindeki `classification` gerekçe listesi **hatalıdır**: "validation eligible 0 < 30" ve "test eligible 0 < 30" uygunluk koşulunun başarısız olduğunu göstermez.
- **Uygun hedef sayısı koşulu sağlanmıştır:** α = 0,10 ve α = 0,20 için doğrulamada 30 (≥ 30) ve testte 56 (≥ 30) uygun A1 hedefi (aynı kaydın `steps.alpha.per_alpha` alanı).
- **A1'in keşifsel sınıflanmasının tek gerçek gerekçesi α kuralının başarısız olmasıdır:** doğrulamada medyan normalize B0 kayması α = 0,10'da 0,0317, α = 0,20'de 0,0476; eşik 0,08.
- Bu sonuçla sıralama deneyinin tamamı keşifsel sınıflandı (protokol bölüm 16).
