export function isSafeEvidenceImageSource(value: unknown): value is string {
  if (typeof value !== 'string') return false
  
  // Eğer data: ile başlıyorsa, kesinlikle sadece JPEG Base64 olmalı
  if (value.startsWith('data:')) {
    const prefix = 'data:image/jpeg;base64,'
    if (!value.startsWith(prefix)) return false
    // Prefix haricinde payload (gerçek base64 verisi) olmalı
    return value.length > prefix.length
  }
  
  // Eğer data: ile başlamıyorsa, backend sadece ham base64 (payload) göndermiş demektir.
  // Boş olmamalı.
  return value.trim().length > 0
}