from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import tempfile
import os
from pydub import AudioSegment

app = FastAPI(title="KnowThis AI Detection API")

# CORS ayarları (iOS uygulaması için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "KnowThis AI Detection API is running"}

@app.post("/analyze")
async def analyze_audio(file: UploadFile):
    """
    Ses dosyasını analiz eder ve AI üretimi olasılığını hesaplar
    """
    if not file.filename.endswith(('.m4a', '.mp3', '.wav', '.aac')):
        raise HTTPException(status_code=400, detail="Desteklenmeyen dosya formatı")
    
    # Geçici dosya oluştur
    with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Önce pydub ile WAV'a çevir (tüm formatları destekler)
        try:
            print(f"📥 Dosya alındı: {file.filename}, boyut: {len(content)} bytes")
            
            # pydub ile oku (m4a, mp3, wav hepsini destekler)
            audio_segment = AudioSegment.from_file(tmp_path)
            
            # Mono yap
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            # Sample rate'i 22050'ye ayarla
            audio_segment = audio_segment.set_frame_rate(22050)
            
            # WAV olarak geçici dosyaya kaydet
            wav_path = tmp_path.replace(tmp_path.split('.')[-1], 'wav')
            audio_segment.export(wav_path, format='wav')
            
            # Librosa ile yükle
            audio, sr = librosa.load(wav_path, sr=22050, duration=10)
            
            # Geçici WAV dosyasını sil
            if os.path.exists(wav_path):
                os.remove(wav_path)
                
        except Exception as e:
            print(f"❌ Audio loading error: {e}")
            # Fallback: direkt librosa ile dene
            audio, sr = librosa.load(tmp_path, sr=22050, duration=10)
        
        # Ses özelliklerini çıkar
        features = extract_audio_features(audio, sr)
        
        # AI detection analizi
        probability, label, reasons = detect_ai_voice(features)
        
        return {
            "probability": float(probability),
            "label": label,
            "reasons": reasons
        }
    
    except Exception as e:
        print(f"❌ Analiz hatası: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")
    
    finally:
        # Geçici dosyayı sil
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def extract_audio_features(audio, sr):
    """
    Ses özelliklerini çıkar
    """
    features = {}
    
    # MFCC (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfcc, axis=1)
    features['mfcc_std'] = np.std(mfcc, axis=1)
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # RMS Energy
    rms = librosa.feature.rms(y=audio)
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    # Pitch
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    
    if pitch_values:
        features['pitch_mean'] = np.mean(pitch_values)
        features['pitch_std'] = np.std(pitch_values)
    else:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
    
    return features

def detect_ai_voice(features):
    """
    Ses özelliklerine göre AI detection yapar
    Gerçek ML modeli yerine kural bazlı basit bir sistem
    """
    reasons = []
    score = 0.5  # Başlangıç skoru
    
    # MFCC analizi - AI sesler genelde daha uniform
    mfcc_variation = np.mean(features['mfcc_std'])
    if mfcc_variation < 15:
        score += 0.15
        reasons.append("MFCC varyasyonu düşük (yapay ses özelliği)")
    elif mfcc_variation > 25:
        score -= 0.1
        reasons.append("MFCC varyasyonu yüksek (doğal ses özelliği)")
    
    # Spectral centroid - AI sesler daha stabil
    if features['spectral_centroid_std'] < 400:
        score += 0.1
        reasons.append("Spektral kararlılık yüksek")
    else:
        reasons.append("Spektral varyasyon doğal seviyede")
    
    # Zero crossing rate - AI seslerde daha düzenli
    if features['zcr_std'] < 0.02:
        score += 0.1
        reasons.append("Ses geçiş oranı çok düzenli")
    
    # RMS energy variation - AI seslerde daha uniform
    if features['rms_std'] < 0.01:
        score += 0.1
        reasons.append("Enerji dağılımı homojen")
    else:
        reasons.append("Enerji dağılımı natural")
    
    # Pitch stability - AI sesler daha stabil pitch
    if features['pitch_std'] < 50 and features['pitch_mean'] > 0:
        score += 0.05
        reasons.append("Perde kararlılığı yüksek")
    
    # Skoru 0-1 arasında tut
    probability = max(0.1, min(0.95, score))
    
    # Label belirle
    if probability < 0.35:
        label = "Düşük"
    elif probability < 0.65:
        label = "Orta"
    else:
        label = "Yüksek"
    
    # En az 3 sebep olsun
    if len(reasons) < 3:
        reasons.append("Genel ses analizi paternleri")
    
    return probability, label, reasons[:3]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
