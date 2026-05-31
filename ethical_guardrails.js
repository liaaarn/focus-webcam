/**
 * FocusWebCam — Ethical Guardrails Module
 * =========================================
 * Memastikan aplikasi memenuhi standar:
 * - Transparansi (explainability)
 * - Mitigasi Bias
 * - Privasi & Security
 * - Regulatory Compliance (GDPR)
 * - Human Intervention / Safety Override
 */

// ─────────────────────────────────────────────
// 1. TRANSPARANSI: Feature Contribution Explanation
// ─────────────────────────────────────────────

class ExplainabilityEngine {
  constructor(modelCoefficients) {
    this.coef = modelCoefficients;
  }

  /**
   * Menghitung kontribusi setiap fitur terhadap prediksi
   * @returns {Object} Penjelasan mengapa skor fokus rendah/tinggi
   */
  explainPrediction(ear, headPose, mouth, probability, score) {
    const contributions = {
      ear: this._normalizeContribution(ear, this.coef.ear, 'positive'),
      head_pose: this._normalizeContribution(headPose, this.coef.head_pose, 'negative'),
      mouth: this._normalizeContribution(mouth, this.coef.mouth_ratio, 'positive')
    };

    // Cari faktor dominan penyebab rendahnya skor
    const negativeFactors = [];
    const positiveFactors = [];

    if (contributions.ear < 0.3) negativeFactors.push('mata tertutup/berkedip berlebihan');
    if (contributions.head_pose < 0.3) negativeFactors.push('kepala menoleh dari layar');
    if (contributions.mouth < 0.3) negativeFactors.push('mulut terbuka (mungkin menguap)');
    
    if (contributions.ear > 0.7) positiveFactors.push('mata terbuka dengan baik');
    if (contributions.head_pose > 0.7) positiveFactors.push('kepala menghadap layar');
    if (contributions.mouth > 0.7) positiveFactors.push('mulut tertutup');

    let explanation = '';
    let suggestion = '';

    if (score < 40) {
      explanation = `Skor fokus rendah (${score}/100). Faktor utama: ${negativeFactors.join(', ')}.`;
      suggestion = this._getSuggestion(negativeFactors);
    } else if (score < 65) {
      explanation = `Skor fokus sedang (${score}/100). ${negativeFactors.length > 0 ? `Perhatikan: ${negativeFactors.join(', ')}.` : 'Pertahankan kondisi saat ini.'}`;
      suggestion = 'Coba kurangi gerakan kepala dan jaga mata tetap fokus.';
    } else {
      explanation = `Skor fokus baik (${score}/100). ${positiveFactors.join(', ')}.`;
      suggestion = 'Pertahankan!';
    }

    return {
      score,
      probability: Math.round(probability * 100),
      explanation,
      suggestion,
      contributions: {
        eye: Math.round(contributions.ear * 100),
        head: Math.round(contributions.head_pose * 100),
        mouth: Math.round(contributions.mouth * 100)
      },
      factors: { negative: negativeFactors, positive: positiveFactors }
    };
  }

  _normalizeContribution(value, coefficient, type) {
    let raw = Math.abs(coefficient) * Math.min(value, 0.5);
    if (type === 'negative' && coefficient < 0) {
      raw = Math.abs(coefficient) * (1 - Math.min(value, 0.3) / 0.3);
    }
    return Math.min(1, Math.max(0, raw));
  }

  _getSuggestion(factors) {
    const suggestions = {
      'mata tertutup/berkedip berlebihan': 'Cobalah lebih sering membuka mata dan kurangi kedipan berlebihan.',
      'kepala menoleh dari layar': 'Posisikan kepala menghadap langsung ke layar kamera.',
      'mulut terbuka (mungkin menguap)': 'Coba regangkan tubuh atau minum air untuk mengurangi rasa kantuk.'
    };
    
    if (factors.length === 0) return 'Terus pertahankan fokus Anda!';
    return suggestions[factors[0]] || 'Jaga posisi tubuh dan kontak mata dengan kamera.';
  }
}

// ─────────────────────────────────────────────
// 2. MITIGASI BIAS: Fairness Validation
// ─────────────────────────────────────────────

class BiasMitigation {
  constructor() {
    this.lightningCompensation = true;
    
    this.fairnessRanges = {
      ear: { min: 0.12, max: 0.38, warning: 'Nilai EAR di luar rentang normal. Pastikan pencahayaan cukup.' },
      head_pose: { min: 0, max: 0.28, warning: 'Deteksi kepala tidak stabil. Periksa posisi wajah.' },
      mouth: { min: 0.001, max: 0.18, warning: 'Deteksi mulut tidak akurat. Periksa pencahayaan.' }
    };
  }

  validateFairness(ear, headPose, mouth) {
    const warnings = [];
    let isCompromised = false;

    if (ear < this.fairnessRanges.ear.min || ear > this.fairnessRanges.ear.max) {
      warnings.push(this.fairnessRanges.ear.warning);
      isCompromised = true;
    }

    if (headPose > this.fairnessRanges.head_pose.max) {
      warnings.push(this.fairnessRanges.head_pose.warning);
    }

    if (mouth > this.fairnessRanges.mouth.max) {
      warnings.push(this.fairnessRanges.mouth.warning);
    }

    return {
      isFair: !isCompromised,
      warnings,
      confidence: isCompromised ? 0.65 : 0.95
    };
  }

  compensateLighting(ear, brightness) {
    if (!this.lightningCompensation) return ear;
    if (brightness < 0.3) {
      return Math.min(ear * 1.15, 0.38);
    }
    if (brightness > 0.8) {
      return Math.max(ear * 0.9, 0.1);
    }
    return ear;
  }

  estimateBrightness(imageData) {
    if (!imageData || !imageData.data) return 0.5;
    let sum = 0;
    for (let i = 0; i < imageData.data.length; i += 4) {
      sum += (imageData.data[i] + imageData.data[i+1] + imageData.data[i+2]) / 3;
    }
    const avg = sum / (imageData.data.length / 4);
    return avg / 255;
  }
}

// ─────────────────────────────────────────────
// 3. PRIVACY & SECURITY: Data Protection (GDPR)
// ─────────────────────────────────────────────

class PrivacyGuard {
  constructor() {
    this.sessionData = [];
    this.consentGiven = false;
    this.retentionPeriod = 24 * 60 * 60 * 1000;
  }

  requestConsent() {
    return new Promise((resolve) => {
      const modal = document.createElement('div');
      modal.className = 'consent-modal';
      modal.innerHTML = `
        <div class="consent-content">
          <h3>📋 Persetujuan Privasi</h3>
          <p>FocusWebCam memproses data wajah Anda untuk mendeteksi tingkat fokus.</p>
          <ul>
            <li>✅ Semua data diproses secara <strong>lokal di perangkat Anda</strong></li>
            <li>✅ Video tidak pernah dikirim ke server manapun</li>
            <li>✅ Data sesi akan dihapus setelah 24 jam</li>
            <li>✅ Anda dapat mengekspor atau menghapus data kapan saja</li>
            <li>✅ Model AI berjalan sepenuhnya di browser Anda</li>
          </ul>
          <div class="consent-buttons">
            <button id="consentAccept" class="btn-consent accept">Izinkan</button>
            <button id="consentReject" class="btn-consent reject">Tolak</button>
          </div>
        </div>
      `;
      
      document.body.appendChild(modal);
      
      document.getElementById('consentAccept').onclick = () => {
        this.consentGiven = true;
        modal.remove();
        resolve(true);
      };
      
      document.getElementById('consentReject').onclick = () => {
        this.consentGiven = false;
        modal.remove();
        resolve(false);
      };
    });
  }

  storeSessionData(data) {
    if (!this.consentGiven) return null;
    
    const anonymizedData = {
      id: crypto.randomUUID ? crypto.randomUUID() : Date.now() + '-' + Math.random(),
      timestamp: Date.now(),
      scores: data.scores.map(s => Math.round(s)),
      avgScore: data.avgScore,
      focusPercentage: data.focusPercentage,
      alertCount: data.alertCount,
      duration: data.duration
    };
    
    this.sessionData.push(anonymizedData);
    this._cleanOldData();
    return anonymizedData.id;
  }

  _cleanOldData() {
    const now = Date.now();
    this.sessionData = this.sessionData.filter(d => 
      now - d.timestamp < this.retentionPeriod
    );
  }

  exportUserData() {
    return {
      exportedAt: new Date().toISOString(),
      appVersion: 'FocusWebCam V2',
      sessions: this.sessionData,
      totalSessions: this.sessionData.length,
      retentionPolicy: '24 hours'
    };
  }

  deleteAllData() {
    this.sessionData = [];
    return true;
  }

  showPrivacyPanel() {
    const panel = document.createElement('div');
    panel.className = 'privacy-panel';
    panel.innerHTML = `
      <div class="privacy-content">
        <h3>🔒 Kontrol Privasi Anda</h3>
        <p><strong>Status:</strong> ${this.consentGiven ? '✅ Persetujuan diberikan' : '⚠️ Belum ada persetujuan'}</p>
        <p><strong>Data tersimpan:</strong> ${this.sessionData.length} sesi</p>
        <p><small>Data disimpan secara lokal di browser Anda dan akan otomatis dihapus setelah 24 jam.</small></p>
        <button id="exportDataBtn" class="btn-privacy">📥 Ekspor Data Saya (JSON)</button>
        <button id="deleteDataBtn" class="btn-privacy danger">🗑️ Hapus Semua Data</button>
        <button id="closePanelBtn" class="btn-privacy">Tutup</button>
      </div>
    `;
    
    document.body.appendChild(panel);
    
    document.getElementById('exportDataBtn').onclick = () => {
      const data = this.exportUserData();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `focuswebcam-data-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    };
    
    document.getElementById('deleteDataBtn').onclick = () => {
      if (confirm('Hapus semua data sesi? Tindakan ini tidak dapat dibatalkan.')) {
        this.deleteAllData();
        alert('Semua data telah dihapus.');
        panel.remove();
      }
    };
    
    document.getElementById('closePanelBtn').onclick = () => panel.remove();
  }
}

// ─────────────────────────────────────────────
// 4. HUMAN INTERVENTION: Safety Override
// ─────────────────────────────────────────────

class SafetyOverride {
  constructor() {
    this.emergencyStop = false;
    this.userOverrides = [];
    this.blinkHistory = [];
    this.lastBlinkFrame = 0;
  }

  detectSafetyHazard(ear, consecutiveFrames, frameCount) {
    const hazards = [];
    
    // Track blink history
    this.blinkHistory.push({ ear, frame: frameCount });
    if (this.blinkHistory.length > 300) this.blinkHistory.shift();
    
    // Hazard 1: Terlalu lama tidak berkedip (EAR tinggi terus)
    const highEarFrames = this.blinkHistory.filter(h => h.ear > 0.32).length;
    if (highEarFrames > 180) {
      hazards.push({
        type: 'eye_strain',
        message: '⚠️ Anda terlalu lama tidak berkedip! Istirahatkan mata sejenak (20-20-20 rule: lihat jauh 20 detik).',
        severity: 'medium'
      });
    }
    
    // Hazard 2: Terlalu banyak menguap (mouth ratio tinggi)
    return hazards;
  }

  requestIntervention(message) {
    const intervention = document.createElement('div');
    intervention.className = 'intervention-overlay';
    intervention.innerHTML = `
      <div class="intervention-card">
        <div class="intervention-icon">🛡️</div>
        <h4>Intervensi Keamanan</h4>
        <p>${message}</p>
        <button class="intervention-dismiss">Saya Mengerti</button>
      </div>
    `;
    
    document.body.appendChild(intervention);
    
    intervention.querySelector('.intervention-dismiss').onclick = () => {
      intervention.remove();
    };
    
    setTimeout(() => {
      if (document.body.contains(intervention)) intervention.remove();
    }, 10000);
  }

  getBlinkRate() {
    if (this.blinkHistory.length < 60) return null;
    // Simple blink detection (EAR drops below threshold then rises)
    let blinkCount = 0;
    let wasLow = false;
    for (let i = 1; i < this.blinkHistory.length; i++) {
      const isLow = this.blinkHistory[i].ear < 0.18;
      if (isLow && !wasLow) blinkCount++;
      wasLow = isLow;
    }
    return (blinkCount / (this.blinkHistory.length / 30)) * 60; // blinks per minute
  }
}

// ─────────────────────────────────────────────
// Export untuk digunakan di focus.js
// ─────────────────────────────────────────────
window.EthicalGuardrails = {
  ExplainabilityEngine,
  BiasMitigation,
  PrivacyGuard,
  SafetyOverride
};