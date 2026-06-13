/**
 * FocusWebCam — Ethical Guardrails Module
 * =========================================
 * Ensures the application meets standards for:
 * - Transparency (explainability)
 * - Bias Mitigation
 * - Privacy & Security
 * - Regulatory Compliance (GDPR)
 * - Human Intervention / Safety Override
 */

// ─────────────────────────────────────────────
// 1. TRANSPARENCY: Feature Contribution Explanation
// ─────────────────────────────────────────────

class ExplainabilityEngine {
  constructor(modelCoefficients) {
    this.coef = modelCoefficients;
  }

  /**
   * Calculates contribution of each feature to the prediction
   * @returns {Object} Explanation of why focus score is low/high
   */
  explainPrediction(ear, headPose, mouth, probability, score) {
    const contributions = {
      ear: this._normalizeContribution(ear, this.coef.ear, "positive"),
      head_pose: this._normalizeContribution(
        headPose,
        this.coef.head_pose,
        "negative",
      ),
      mouth: this._normalizeContribution(
        mouth,
        this.coef.mouth_ratio,
        "positive",
      ),
    };

    // Find dominant factors causing low score
    const negativeFactors = [];
    const positiveFactors = [];

    if (contributions.ear < 0.3)
      negativeFactors.push("eyes closed/excessive blinking");
    if (contributions.head_pose < 0.3)
      negativeFactors.push("head turned away from screen");
    if (contributions.mouth < 0.3)
      negativeFactors.push("mouth open (possibly yawning)");

    if (contributions.ear > 0.7)
      positiveFactors.push("eyes open well");
    if (contributions.head_pose > 0.7)
      positiveFactors.push("head facing screen");
    if (contributions.mouth > 0.7) positiveFactors.push("mouth closed");

    let explanation = "";
    let suggestion = "";

    if (score < 40) {
      explanation = `Low focus score (${score}/100). Main factors: ${negativeFactors.join(", ")}.`;
      suggestion = this._getSuggestion(negativeFactors);
    } else if (score < 65) {
      explanation = `Moderate focus score (${score}/100). ${negativeFactors.length > 0 ? `Pay attention to: ${negativeFactors.join(", ")}.` : "Maintain current condition."}`;
      suggestion = "Try to reduce head movements and keep eyes focused.";
    } else {
      explanation = `Good focus score (${score}/100). ${positiveFactors.join(", ")}.`;
      suggestion = "Keep it up!";
    }

    return {
      score,
      probability: Math.round(probability * 100),
      explanation,
      suggestion,
      contributions: {
        eye: Math.round(contributions.ear * 100),
        head: Math.round(contributions.head_pose * 100),
        mouth: Math.round(contributions.mouth * 100),
      },
      factors: { negative: negativeFactors, positive: positiveFactors },
    };
  }

  _normalizeContribution(value, coefficient, type) {
    let raw = Math.abs(coefficient) * Math.min(value, 0.5);
    if (type === "negative" && coefficient < 0) {
      raw = Math.abs(coefficient) * (1 - Math.min(value, 0.3) / 0.3);
    }
    return Math.min(1, Math.max(0, raw));
  }

  _getSuggestion(factors) {
    const suggestions = {
      "eyes closed/excessive blinking":
        "Try to keep your eyes open more often and reduce excessive blinking.",
      "head turned away from screen":
        "Position your head directly facing the camera screen.",
      "mouth open (possibly yawning)":
        "Try stretching or drinking water to reduce drowsiness.",
    };

    if (factors.length === 0) return "Keep maintaining your focus!";
    return (
      suggestions[factors[0]] ||
      "Maintain body posture and eye contact with the camera."
    );
  }
}

// ─────────────────────────────────────────────
// 2. BIAS MITIGATION: Fairness Validation
// ─────────────────────────────────────────────

class BiasMitigation {
  constructor() {
    this.lightningCompensation = true;

    this.fairnessRanges = {
      ear: {
        min: 0.12,
        max: 0.38,
        warning:
          "EAR value outside normal range. Ensure adequate lighting.",
      },
      head_pose: {
        min: 0,
        max: 0.28,
        warning: "Head detection unstable. Check face position.",
      },
      mouth: {
        min: 0.001,
        max: 0.18,
        warning: "Mouth detection inaccurate. Check lighting.",
      },
    };
  }

  validateFairness(ear, headPose, mouth) {
    const warnings = [];
    let isCompromised = false;

    if (
      ear < this.fairnessRanges.ear.min ||
      ear > this.fairnessRanges.ear.max
    ) {
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
      confidence: isCompromised ? 0.65 : 0.95,
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
      sum +=
        (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
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
      const modal = document.createElement("div");
      modal.className = "consent-modal";
      modal.innerHTML = `
        <div class="consent-content">
          <h3>📋 Privacy Consent</h3>
          <p>FocusWebCam processes your facial data to detect focus levels.</p>
          <ul>
            <li>✅ All data is processed <strong>locally on your device</strong></li>
            <li>✅ Video is never sent to any server</li>
            <li>✅ Session data will be deleted after 24 hours</li>
            <li>✅ You can export or delete your data at any time</li>
            <li>✅ AI model runs entirely in your browser</li>
          </ul>
          <div class="consent-buttons">
            <button id="consentAccept" class="btn-consent accept">Allow</button>
            <button id="consentReject" class="btn-consent reject">Deny</button>
          </div>
        </div>
      `;

      document.body.appendChild(modal);

      document.getElementById("consentAccept").onclick = () => {
        this.consentGiven = true;
        modal.remove();
        resolve(true);
      };

      document.getElementById("consentReject").onclick = () => {
        this.consentGiven = false;
        modal.remove();
        resolve(false);
      };
    });
  }

  storeSessionData(data) {
    if (!this.consentGiven) return null;

    const anonymizedData = {
      id: crypto.randomUUID
        ? crypto.randomUUID()
        : Date.now() + "-" + Math.random(),
      timestamp: Date.now(),
      scores: data.scores.map((s) => Math.round(s)),
      avgScore: data.avgScore,
      focusPercentage: data.focusPercentage,
      alertCount: data.alertCount,
      duration: data.duration,
    };

    this.sessionData.push(anonymizedData);
    this._cleanOldData();
    return anonymizedData.id;
  }

  _cleanOldData() {
    const now = Date.now();
    this.sessionData = this.sessionData.filter(
      (d) => now - d.timestamp < this.retentionPeriod,
    );
  }

  exportUserData() {
    return {
      exportedAt: new Date().toISOString(),
      appVersion: "FocusWebCam V2",
      sessions: this.sessionData,
      totalSessions: this.sessionData.length,
      retentionPolicy: "24 hours",
    };
  }

  deleteAllData() {
    this.sessionData = [];
    return true;
  }

  showPrivacyPanel() {
    const panel = document.createElement("div");
    panel.className = "privacy-panel";
    panel.innerHTML = `
      <div class="privacy-content">
        <h3>🔒 Your Privacy Control</h3>
        <p><strong>Status:</strong> ${this.consentGiven ? "✅ Consent given" : "⚠️ No consent yet"}</p>
        <p><strong>Stored data:</strong> ${this.sessionData.length} sessions</p>
        <p><small>Data is stored locally in your browser and automatically deleted after 24 hours.</small></p>
        <button id="exportDataBtn" class="btn-privacy">📥 Export My Data (JSON)</button>
        <button id="deleteDataBtn" class="btn-privacy danger">🗑️ Delete All Data</button>
        <button id="closePanelBtn" class="btn-privacy">Close</button>
      </div>
    `;

    document.body.appendChild(panel);

    document.getElementById("exportDataBtn").onclick = () => {
      const data = this.exportUserData();
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `focuswebcam-data-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    };

    document.getElementById("deleteDataBtn").onclick = () => {
      if (
        confirm("Delete all session data? This action cannot be undone.")
      ) {
        this.deleteAllData();
        alert("All data has been deleted.");
        panel.remove();
      }
    };

    document.getElementById("closePanelBtn").onclick = () => panel.remove();
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

    // Hazard 1: Too long without blinking (consistently high EAR)
    const highEarFrames = this.blinkHistory.filter((h) => h.ear > 0.32).length;
    if (highEarFrames > 180) {
      hazards.push({
        type: "eye_strain",
        message:
          "⚠️ You haven't blinked in a while! Rest your eyes (20-20-20 rule: look 20 feet away for 20 seconds).",
        severity: "medium",
      });
    }

    // Hazard 2: Excessive yawning (high mouth ratio)
    return hazards;
  }

  requestIntervention(message) {
    const intervention = document.createElement("div");
    intervention.className = "intervention-overlay";
    intervention.innerHTML = `
      <div class="intervention-card">
        <div class="intervention-icon">🛡️</div>
        <h4>Safety Intervention</h4>
        <p>${message}</p>
        <button class="intervention-dismiss">I Understand</button>
      </div>
    `;

    document.body.appendChild(intervention);

    intervention.querySelector(".intervention-dismiss").onclick = () => {
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
// Export for use in focus.js
// ─────────────────────────────────────────────
window.EthicalGuardrails = {
  ExplainabilityEngine,
  BiasMitigation,
  PrivacyGuard,
  SafetyOverride,
};