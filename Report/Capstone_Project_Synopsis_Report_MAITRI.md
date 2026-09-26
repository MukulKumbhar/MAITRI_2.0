# Capstone Project Synopsis on
# MAITRI: Multimodal AI-based Astronaut Telemetry & Real-time Intelligence
### *(Multimodal Astronaut Psychological Well-being and Emotional Support System)*

**Submitted in partial fulfillment of the requirements for the Capstone Project of**

### **BACHELOR OF TECHNOLOGY**
in
### **AERONAUTICAL ENGINEERING**
*(Minor in Aerospace Engineering / Department of Computer Science & Engineering)*

<br>

**Submitted by**

| Student Name | Roll No / URN | Branch & Specialization |
| :--- | :---: | :--- |
| **Mr. MUKUL RAMESH KUMBHAR** | **23031012** | B.Tech — Computer Science & Engineering *(Lead Systems Architect & ML Pipeline)* |
| **Mr. ADITYA ANANDA KUMBHAR** | **23031016** | B.Tech — Computer Science & Engineering *(Facial Affect & Video Pipeline Lead)* |
| **Mr. ARYAN AMARSINH PATIL** | **23101058** | B.Tech — CSE (Internet of Things) *(Physiological Telemetry & Vitals Lead)* |
| **Mr. SHREEDHAR SURESH DALVI** | **23041084** | B.Tech — Electrical Engineering *(Eye Tracking & Sensor Interfaces Lead)* |
| **Ms. PRANJAL SAMPATRAO MANE** | **23101007** | B.Tech — CSE (Internet of Things) *(Multimodal Fusion & Database Analytics Lead)* |

<br>

**Under the Guidance of**

**Dr. SAKTHIPRIYA BALU**  
*Assistant Professor & Project Supervisor*  
Department of Aeronautical Engineering

<br>

### **DEPARTMENT OF AERONAUTICAL ENGINEERING**
**Sant Dnyaneshwar Shikshan Sanstha's**

## **ANNASAHEB DANGE COLLEGE OF ENGINEERING AND TECHNOLOGY, ASHTA**
**(An Empowered Autonomous Institute)**
### **SHIVAJI UNIVERSITY, KOLHAPUR**
**Academic Year: 2025–2026**

<br>

---

\newpage

# Sponsorship Certificate / Certificate of Institutional Approval

This is to certify that the Capstone Project Synopsis entitled:

> **"MAITRI: Multimodal AI-based Astronaut Telemetry & Real-time Intelligence (Multimodal Astronaut Psychological Well-being and Emotional Support System)"**

is a bonafide proposal submitted by:

* **Mr. MUKUL RAMESH KUMBHAR** (URN: 23031012)
* **Mr. ADITYA ANANDA KUMBHAR** (URN: 23031016)
* **Mr. ARYAN AMARSINH PATIL** (URN: 23101058)
* **Mr. SHREEDHAR SURESH DALVI** (URN: 23041084)
* **Ms. PRANJAL SAMPATRAO MANE** (URN: 23101007)

in partial fulfillment of the requirements for the award of the degree of **Bachelor of Technology in Aeronautical Engineering (Minor in Aerospace Engineering)** at **Annasaheb Dange College of Engineering and Technology (ADCET), Ashta**, affiliated to **Shivaji University, Kolhapur**, during the academic year 2025–2026.

The project proposal has been evaluated and approved in accordance with ADCET Capstone Project Regulations (*Ref: ADCET/Aero/Project/Regulations-2025, Rev-0, Dated 01/01/2025*).

<br><br><br>

```text
       _________________________________                _________________________________
             Dr. SAKTHIPRIYA BALU                              Dr. AMOL S. DANGE
             Project Supervisor / Guide                    Capstone Project Coordinator
       Dept. of Aeronautical Engineering                 Dept. of Aeronautical Engineering
```

<br><br><br>

```text
       _________________________________                _________________________________
              Head of Department                                    Director
       Dept. of Aeronautical Engineering                          ADCET, Ashta
          ADCET, Ashta (Kolhapur)                         (Autonomous Institute)
```

<br>

---

\newpage

# Abstract

During Long-Duration Spaceflight (LDSF) missions—such as deep-space lunar transit, Martian orbital habitats, and planetary surface expeditions—astronaut crews are exposed to persistent, extreme physiological and psychological stressors. Microgravity-induced cephalic fluid shifts, confinement within restricted habitable volumes, monotonous nutrition, circadian rhythm disruption from artificial lighting, and high-consequence operational workloads collectively induce acute cognitive fatigue, affective blunting, and psychosomatic strain. Compounding this challenge, the vast interplanetary distances introduce electromagnetic communication latencies ranging from 3 to 22 minutes each way (resulting in up to 44 minutes roundtrip delay for Mars), completely eliminating the possibility of real-time terrestrial tele-counseling from Earth-based flight psychologists. Furthermore, operational spaceflight culture frequently encourages astronauts to suppress subjective distress, allowing acute psychological degradation to proceed unnoticed until it precipitates catastrophic operational errors.

To address this aerospace healthcare challenge, this project develops **MAITRI (Multimodal AI-based Astronaut Telemetry & Real-time Intelligence)**, an autonomous, completely offline, edge-native multimodal monitoring and closed-loop affective support architecture. The system is architected into two clearly defined development phases:
1. **Phase 1 (Existing Implemented Foundation):** A high-throughput, decoupled multimodal monitoring subsystem operating locally at 30 FPS without cloud dependency. Upgraded from an initial high-latency VGG-Face baseline, Phase 1 integrates an optimized EfficientNet-B2 ONNX model (`enet_b2_7.onnx`) running in ~6.3 ms for Facial Emotion Recognition (FER), a 478-point 3D landmark mesh (MediaPipe Tasks) computing Eye Aspect Ratio (EAR) and rolling 60-second blink rates with reading-stupor hysteresis, a Digital Audio Processing (DAP) pipeline with a 75 Hz infrasonic high-pass filter that eliminates spacecraft life-support cooling fan noise (ALC257 rumble), a heuristic laughter/sobbing reflex bypass ($R_{\text{env}} \sim 0.90$), and an SNR-weighted dual-model Speech Emotion Recognition (SER) ensemble (Wav2Vec2 ONNX + synthetically trained MFCC-MLP ONNX). These streams are fused via a dynamic, Quality-Gated Exponential Moving Average (EMA, $\alpha=0.15$) engine that calculates a unified Stress Index ($0-100\%$), incorporates physical exertion cross-validation, and drives an inactivity watchdog, Delay-Tolerant Networking (DTN) ground outbox simulator, and SQLite telemetry database (`maitri_logs.db`).
2. **Phase 2 (Proposed Closed-Loop Support Development):** A safety-gated, closed-loop, offline affective support architecture operating under the operational paradigm **Sense $\to$ Assess $\to$ Select $\to$ Support $\to$ Reassess $\to$ Adapt**. Phase 2 consumes structured 1 Hz telemetry snapshots via a read-only State Gateway, filters transient emotional noise through temporal persistence slopes, and drives a deterministic 6-State Finite State Machine (FSM) governed by dwell-time hysteresis (15s) and refractory cooldown timers (180s) to eliminate intervention flapping. Non-pharmacological countermeasures (Box Breathing 4-4-4-4, 4-7-8 parasympathetic regulation, 5-4-3-2-1 sensory grounding) are administered via a retrieval-grounded local knowledge base, a quantized open-weights conversational LLM bounded by strict pre- and post-generation safety gates, offline speech I/O (`faster-whisper` + Piper TTS), and tempo-regulated psychoacoustic adaptive audio (60–70 BPM). A closed-loop feedback engine evaluates post-intervention affective differentials ($\Delta \text{Stress}$) and discrete astronaut feedback to adapt support policies or deterministically escalate emergencies to ground control.

**Keywords:** Multimodal Affective Computing, Astronaut Psychological Well-being, Delay-Tolerant Networking (DTN), Edge AI, Closed-Loop Behavioral Countermeasures, Finite State Machine, Cognitive Behavioral Therapy (CBT), Inactivity Watchdog.

---

\newpage

# Contents

- **Abstract** .......................................................................................................................... i
- **Sponsorship Certificate** ..................................................................................................... ii
- **1 Introduction** ................................................................................................................. 1
  - 1.1 Background and Context ......................................................................................... 1
  - 1.2 Purpose .................................................................................................................... 1
- **2 Literature Survey** ........................................................................................................ 2
  - Tabular Comparative Analysis of Prior Art ...................................................................... 2
  - Observation and Research Gap ...................................................................................... 3
- **3 Problem Statement** ..................................................................................................... 4
- **4 Objectives** ................................................................................................................... 5
  - 4.1 Phase 1 Implemented Objectives (Monitoring & Telemetry) ........................................ 5
  - 4.2 Phase 2 Proposed Objectives (Closed-Loop Affective Support) .................................... 5
- **5 Scope** ......................................................................................................................... 6
  - 5.1 Included Technical Scope ......................................................................................... 6
  - 5.2 Excluded Scope and Engineering Limitations ............................................................ 6
- **6 Proposed Work** ........................................................................................................... 7
  - 6.1 Methodology (Proposed Modules, Techniques, Algorithms, Architecture) ................. 7
    - System Architecture and Conceptual Flow .................................................................... 7
    - Phase 1 Implemented Multimodal Pipeline .................................................................... 8
    - Phase 2 Proposed Closed-Loop Support Architecture ................................................... 11
    - Core Subsystem Modules (M1 to M10) ......................................................................... 14
    - Techniques and Algorithms .......................................................................................... 16
  - 6.2 Software and Hardware Requirements and Availability ............................................... 20
    - Software Requirements ............................................................................................... 20
    - Hardware Requirements ............................................................................................... 21
    - Availability .................................................................................................................. 21
- **7 Expected Outcome** .................................................................................................... 22
  - 7.1 Verified Phase 1 Outcomes ..................................................................................... 22
  - 7.2 Projected Phase 2 Outcomes ................................................................................... 22
- **8 Schedule** .................................................................................................................... 23
- **Expected Challenges, Risks, and Mitigation Strategies** ................................................. 24
- **References** .................................................................................................................... 25
- **Details of Team Members and Signatures** ...................................................................... 27
- **Annexure - I: ADCET Capstone Project Proposal (Proforma - I)** ..................................... 28

---

\newpage

# 1 Introduction

Astronaut psychological health and cognitive readiness represent mission-critical determinants of success in human space exploration. During long-duration missions aboard orbital space stations or deep-space transit vehicles, crew members endure prolonged sensory deprivation, chronic isolation, high-stress operational schedules, and severe environmental confinement. Traditional mission operations have relied upon real-time tele-counseling with Earth-based flight surgeons. However, beyond Low Earth Orbit, transmission delays and communication blackouts render continuous Earth-dependent psychiatric support impossible.

With the advent of high-performance edge artificial intelligence and multimodal perception algorithms, automated systems can now monitor human physiological and psychological states locally. By embedding edge computing directly into spacecraft habitat avionics, it becomes possible to continuously assess astronaut well-being, eliminate single-sensor ambiguities, and deploy autonomous behavioral countermeasures without requiring cloud connectivity.

The **MAITRI (Multimodal AI-based Astronaut Telemetry & Real-time Intelligence)** system is designed to provide an autonomous, privacy-preserving, edge-native solution for monitoring and supporting astronaut psychological well-being. Operating 100% offline, MAITRI fuses computer vision, speech acoustics, and vital telemetry into an objective stress index and couples it with a deterministic, safety-gated support policy engine that delivers adaptive, closed-loop psychological countermeasures.

## 1.1 Background and Context
In deep-space exploration missions (such as ISRO Gaganyaan/BAS, NASA Artemis, and crewed Mars expeditions), crews face physical and psychological isolation unprecedented in human history. Astronauts live and work inside tightly confined pressurized volumes subjected to constant acoustic hum from Environmental Control and Life Support Systems (ECLSS), microgravity-induced cephalic fluid shifts, circadian desynchronization, and the profound psychological phenomenon known as the "Earth-out-of-view" effect.

Under terrestrial conditions, private psychological conferences (PPCs) occur regularly between astronauts and terrestrial operational psychologists. However, for interplanetary transit to Mars, the one-way speed-of-light propagation delay varies between 3 and 22 minutes ($\tau_{\text{delay}} = d_{\text{propagation}} / c$). This latency produces a minimum roundtrip conversational lag of 6 to 44 minutes, completely precluding interactive tele-counseling or crisis de-escalation. Furthermore, during solar conjunction, communication links between Earth and Mars are completely severed for up to several weeks due to solar radio interference.

In addition to physical communication constraints, operational spaceflight culture frequently encourages crew members to suppress subjective fatigue, anxiety, or interpersonal friction to maintain flight status. Retrospective analyses from terrestrial isolation analog studies (such as the 520-day Mars500 experiment) confirm that psychological degradation manifests insidiously, often undetected until cognitive failure impairs mission safety. Consequently, space agencies have identified autonomous medical decision support as a high-priority technology gap.

## 1.2 Purpose
The main purpose of the **MAITRI** system is to develop an edge-native, zero-cloud, multimodal monitoring and closed-loop affective support system that maintains astronaut psychological well-being and operational performance during long-duration space missions.

The system is intended to:
- Continuously and non-invasively monitor astronaut affective states, eye fatigue, vocal prosody, and physiological vitals using standard cabin sensors.
- Synthesize disparate sensory inputs into a standardized, anti-flicker Stress Index (0–100%) governed by dynamic quality gating and temporal smoothing.
- Disambiguate healthy operational activities from true psychological distress through context-aware cross-validation (e.g., distinguishing treadmill exercise tachycardia from acute panic, and quiet manual reading from catatonic stupor).
- Administer evidence-based, non-pharmacological behavioral countermeasures (Cognitive Behavioral Therapy protocols, paced somatic breathing, and adaptive audio) via a deterministic Finite State Machine (FSM).
- Enforce strict deterministic safety gating around conversational AI components, ensuring that emergency escalations bypass generative models.
- Close the loop through post-intervention affective reassessment, verifying stabilization and adapting subsequent support policies.
- Safeguard mission telemetry through an embedded SQLite mission database and a Delay-Tolerant Networking (DTN) ground outbox simulator that queues priority alerts for Earth synchronization.

---

\newpage

# 2 Literature Survey

| Year | System / Paper | Methodology | Advantages | Limitations | Relevance to MAITRI |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1997** | Picard, R. W. *(MIT Press)* | Foundational *Affective Computing* theory; mapping emotions via autonomic biosignals (GSR, HR, EMG) and facial markers. | Established formal mathematical framework for affective modeling and human-machine emotional interaction. | Lacked real-time video/audio processing capacity; relied on obtrusive wired sensors. | Provides theoretical grounding for MAITRI's multi-sensor emotional inference and physiological strain scaling. |
| **2007** | Dinges, D. F., et al. *(NASA/CR-2007-214811)* | Optical Computer Recognition (OCR) of facial expressions using FACS Action Units to track stress and fatigue during analog space tasks. | Demonstrated that optical tracking of high-dimensional facial action units reliably tracks acute stress without electrodes. | Computationally intensive; highly sensitive to cabin lighting and head pose; lacked an active intervention loop. | Directly inspires MAITRI's optical affect tracking, highlighting the necessity of Digital Image Processing (DIP) preprocessing. |
| **2014** | Basner, M., et al. *(PLoS ONE)* | Behavioral and psychological assessment during 520-day Mars analog confinement (Mars500) using vigilance tests and self-reports. | Proved psychological degradation in LDSF is heterogeneous (hypokinesis, depressive symptoms) and progresses insidiously. | Retrospective analysis relying primarily on self-reported questionnaires; lacked real-time autonomous closed-loop mitigation. | Validates the mission necessity of MAITRI's continuous multi-factor monitoring and objective inactivity watchdog. |
| **2016** | Soukupova, T., & Cech, J. *(CVWW)* | Real-Time Eye Blink Detection using Facial Landmarks and the Eye Aspect Ratio (EAR) metric derived from 2D camera frames. | Highly efficient, real-time geometric scalar (<1 ms computation); invariant to uniform scale and minor head rotations. | Static thresholds trigger false alarms during tasks involving variable gaze angles or downward reading postures. | Forms the algorithmic baseline for MAITRI's Module 2, extended with dynamic baseline calibration and reading hysteresis. |
| **2018** | Toups, P. O., et al. *(NASA Tech Memo)* | Autonomous Medical Decision Support (AMDS) concepts for exploration-class space missions, formalizing clinical guidance autonomy. | Defined operational tiers for medical autonomy under extreme communication latency. | Architectural specification without an integrated live multimodal software implementation or affective intervention pipeline. | Serves as the high-level aerospace operational guideline for MAITRI's deterministic emergency outbox and flight surgeon dossiers. |
| **2020** | Baevski, A., et al. *(NeurIPS)* | Wav2Vec 2.0: Self-supervised learning of speech representations directly from raw audio waveforms, capturing rich prosody and phonetics. | State-of-the-art representations for vocal emotion classification; robust to linguistic variations. | High memory footprint (~360MB fp32); sensitive to low-frequency background hum (ALC257 fan noise) causing classification drift. | Serves as the primary acoustic transformer in MAITRI's dual-model SER, mitigated by DAP pre-filtering and MFCC-MLP ensemble. |
| **2022** | Chlan, L. L., et al. *(Intensive Care Nurs)* | Non-pharmacological music and auditory pacing interventions for anxiety and autonomic nervous system regulation in critical care patients. | Validated structured acoustic pacing (tempo <70 BPM) significantly decreases cortisol and heart rate without drug side-effects. | Manual therapist administration; lacked real-time physiological feedback loops to modulate sound dynamically. | Directly justifies MAITRI's Phase 2 Adaptive Audio countermeasure, grounding audio selection in physiological stabilization rather than pseudoscience. |

### Observation and Research Gap:
Most existing aerospace monitoring systems operate strictly as passive diagnostic tools that log stress for post-hoc analysis by terrestrial medical officers. Under deep-space communication delays (>= 22 min), passive monitoring is fundamentally inadequate; the system must execute an immediate, closed-loop behavioral countermeasure. Furthermore, cloud-dependent machine learning is impossible without internet connectivity, while unconstrained generative LLMs introduce dangerous hallucination risks. Therefore, a 100% offline, low-latency multimodal monitoring pipeline coupled with a deterministic, safety-gated closed-loop support engine is required for deep-space missions.

---

\newpage

# 3 Problem Statement

- **IDEAL:** Astronaut crews on deep-space exploration missions should possess continuous, objective, and non-invasive psycho-physiological monitoring capable of detecting mental fatigue and distress at the earliest onset, supported by real-time private psychiatric counseling from Earth and automated onboard behavioral countermeasures to sustain peak cognitive and operational performance.

- **REALITY:** In long-duration deep-space transit (such as missions to Mars or lunar outposts), electromagnetic wave propagation introduces light-speed communication delays of 3 to 22 minutes each way (up to 44 minutes roundtrip), rendering real-time tele-counseling physically impossible. Deep-space telemetry links suffer from strict bandwidth limitations, antenna shadowing, and multi-week solar conjunction blackouts. Simultaneously, astronauts operate inside noisy, confined habitats, enduring microgravity fluid shifts and circadian disruption, while routinely suppressing subjective symptoms of distress due to operational spaceflight culture. Furthermore, spacecraft avionics possess strictly limited edge computational resources that cannot host massive, unoptimized cloud AI models.

- **CONSEQUENCES:**
  - Unnoticed accumulation of acute stress and cognitive fatigue leading to attentional tunneling, degraded psychomotor vigilance, and catastrophic operational errors during safety-critical maneuvers (e.g., orbital docking, EVA operations).
  - Escalation of unmitigated anxiety into acute panic attacks or dissociative stupors without immediate therapeutic support.
  - Development of chronic psychosomatic exhaustion, sleep pathology, and interpersonal friction among crew members, threatening mission cohesion.
  - Alarm fatigue and system deactivation caused by naive single-sensor monitors that trigger false alarms during routine activities (e.g., misinterpreting exercise tachycardia as panic, or reading posture as catatonia).
  - Complete operational isolation of the crew during solar conjunction communication blackouts without autonomous behavioral support.

- **PROPOSAL:** To develop **MAITRI (Multimodal AI-based Astronaut Telemetry & Real-time Intelligence)**, an autonomous, 100% offline, edge-native software architecture that synthesizes live facial affect, eye fatigue, acoustic speech prosody, and physiological vitals into an anti-flicker Stress Index (0–100%) via quality-gated multimodal fusion, and drives a deterministic, safety-gated closed-loop affective support engine featuring structured CBT countermeasures, a local quantized LLM, tempo-regulated adaptive audio, and Delay-Tolerant Networking (DTN) ground escalation.

---

\newpage

# 4 Objectives

## 4.1 Phase 1 Implemented Objectives (Monitoring & Telemetry)
- To develop a decoupled, sub-10ms facial affect classification pipeline utilizing an optimized EfficientNet-B2 ONNX model (`enet_b2_7.onnx`) running on ONNX Runtime with 6 intra-op threads to classify 7 canonical emotions (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) at 30 FPS.
- To implement 478-point 3D facial landmark tracking (MediaPipe Tasks API) to compute Eye Aspect Ratio (EAR), track rolling 60-second blink rates via rising-edge hysteresis, and compute clinical PERCLOS fatigue states with adaptive reading-stupor hysteresis (lowering threshold to 5 blinks/min).
- To engineer a Digital Audio Processing (DAP) pipeline featuring a 75 Hz FFT infrasonic high-pass filter to eliminate spacecraft cooling fan rumble (ALC257 noise), a heuristic laughter/sobbing reflex bypass ($R_{\text{env}} \sim 0.90$), and an SNR-weighted dual-model Speech Emotion Recognition (SER) ensemble (Wav2Vec2 ONNX + synthetically trained MFCC-MLP ONNX).
- To formulate a piecewise continuous mathematical model converting Heart Rate (50–160 BPM), Skin Temperature (35–40°C), and Blood Oxygen Saturation ($SpO_2$: 85–100%) into a normalized Physiological Strain Index ($0.0-1.0$) with non-linear penalties for hypoxia and extreme tachycardia.
- To design a dynamic, Quality-Gated Exponential Moving Average (EMA, $\alpha=0.15$) Multimodal Fusion Engine that dynamically reweights inputs during camera occlusion or acoustic silence, incorporating clinical cross-validation rules to prevent false panic alarms during astronaut exercise.
- To construct a zero-lag WebRTC streaming pipeline operating at 30 FPS with thread-safe shared state synchronization (`live_state.py`) and a fixed-scale 480p Heads-Up Display (HUD) video overlay.
- To integrate a Multi-Factor Inactivity Watchdog (frame-to-frame velocity tracking with 8-pixel shift threshold), an embedded SQLite mission database (`maitri_logs.db`), an automated Markdown Flight Surgeon Dossier generator, and a Delay-Tolerant Networking (DTN) simulator modeling deep-space propagation delays.

## 4.2 Phase 2 Proposed Objectives (Closed-Loop Affective Support)
- To construct a decoupled, thread-safe State Gateway providing a 1 Hz read-only structured JSON snapshot interface between Phase 1 and Phase 2 without consuming raw video frames or degrading monitoring throughput.
- To develop a temporal persistence and trend slope evaluation algorithm that verifies sustained affective strain ($\Delta t_{\text{persist}} \ge 12.0$s) and task context, eliminating transient emotional noise.
- To implement a deterministic Support Policy Engine governed by a 6-State Finite State Machine (`STABLE`, `MILD_CONCERN`, `SUPPORT`, `DE_ESCALATION`, `RECOVERY`, `REASSESS`) augmented with dwell-time hysteresis (15s) and refractory cooldown timers (180s) to eliminate intervention flapping.
- To compile an offline, structured psychological knowledge base containing validated Cognitive Behavioral Therapy (CBT) protocols, Box Breathing (4-4-4-4), 4-7-8 Parasympathetic regulation, 5-4-3-2-1 Sensory Grounding, and mission contingency checklists.
- To deploy and benchmark a quantized 4-bit open-weights conversational LLM (Llama-3-8B-Instruct Q4, Mistral-7B Q4, Phi-3-mini Q4) dedicated strictly to natural-language dialogue, strictly prohibited from altering clinical or escalation decisions.
- To integrate an offline speech interface combining `faster-whisper` (with Silero VAD) for low-latency speech-to-text with Piper neural TTS for local auditory guidance.
- To engineer a psychoacoustic Adaptive Audio Engine dynamically selecting tempo-regulated instrumental tracks (60–70 BPM) and biophilic soundscapes to downregulate sympathetic arousal, automatically muted during critical operations.
- To establish a closed-loop feedback engine computing objective affective differentials ($\Delta \text{Stress} = \text{Stress}_{\text{post}} - \text{Stress}_{\text{pre}}$) and integrating discrete astronaut self-reports (`Better`, `Same`, `Worse`) to iteratively adapt support policies.
- To enforce dual deterministic safety gates (Pre-LLM Risk Filter and Post-LLM Output Validator) that intercept clinical emergencies and divert critical events directly to cabin alarms and the DTN outbox.

---

\newpage

# 5 Scope

## 5.1 Included Technical Scope
- Multimodal non-invasive monitoring (2D video, microphone audio, physiological telemetry).
- Sub-10ms facial affect classification (EfficientNet-B2 ONNX) and eye fatigue tracking (MediaPipe 478-pt EAR/PERCLOS).
- Dual-model speech emotion recognition (Wav2Vec2 + MFCC-MLP) with 75 Hz infrasonic high-pass filtering and heuristic reflex overrides.
- Physiological vitals strain modeling (HR, Skin Temp, SpO2 piecewise linear scaling).
- Quality-aware EMA multimodal fusion (alpha=0.15) with context-aware exercise cross-validation.
- Multi-factor inactivity watchdog tracking frame-to-frame velocity and blinks.
- Delay-Tolerant Networking (DTN) ground outbox simulation and automated Markdown Flight Surgeon Dossier generation.
- Decoupled 1 Hz State Gateway providing structured JSON snapshots to Phase 2.
- Deterministic 6-state FSM support policy with dwell-time hysteresis and refractory cooldown timers.
- Local offline psychological knowledge base (Box Breathing, 4-7-8 regulation, 5-4-3-2-1 Grounding).
- Quantized local conversational LLM bounded by dual deterministic safety gates.
- Offline speech I/O via faster-whisper and Piper neural TTS.
- Psychoacoustic adaptive audio pacing (60–70 BPM) and biophilic soundscapes.
- Closed-loop feedback delta engine evaluating Delta_Stress and astronaut self-assessments.
- 100% offline, zero-cloud execution with local SQLite telemetry logging (maitri_logs.db).

## 5.2 Excluded Scope and Engineering Limitations
- Not an autonomous clinical psychiatric authority; does not provide formal medical diagnoses (e.g., DSM-5 depression/PTSD) or pharmacological treatment.
- Does not replace human flight surgeons or operational psychologists.
- No unconstrained generative AI medical decision-making; critical alerts strictly follow deterministic FSM rules.
- No physical deep-space RF hardware (Deep Space Network transponders); DTN delays and packet bundling are software-simulated.
- No unverified acoustic pseudoscience (strictly rejects 432 Hz / 528 Hz claims; grounded in psychoacoustic tempo pacing).
- No formal NASA flight qualification or FDA clinical certification claims (targeted as a proof-of-concept capstone at TRL 4–5).

---

\newpage

# 6 Proposed Work

## 6.1 Methodology (Proposed Modules, Techniques, Algorithms, Architecture)

```text
====================================================================================================
                        MAITRI: INTEGRATED DUAL-PHASE SYSTEM ARCHITECTURE
====================================================================================================

      CABIN SENSORS (100% Offline / Non-Invasive)
      ┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
      │ Live HD Video Stream  │   │ Cabin Audio Stream    │   │ Physiological Vitals  │
      │ (WebRTC 30 FPS)       │   │ (16 kHz Ring Buffer)  │   │ (HR, Temp, SpO2)      │
      └───────────┬───────────┘   └───────────┬───────────┘   └───────────┬───────────┘
                  │                           │                           │
  ================│===========================│===========================│=========================
  PHASE 1: MULTIMODAL MONITORING & TELEMETRY FOUNDATION (IMPLEMENTED)     │
  ================│===========================│===========================│=========================
                  ▼                           ▼                           ▼
      ┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
      │ DIP Enhancement       │   │ DAP Infrasonic Filter │   │ Piecewise Mathematical│
      │ CLAHE, Gamma, Unsharp │   │ 75Hz FFT High-Pass    │   │ Physiological Strain  │
      └───────────┬───────────┘   └───────────┬───────────┘   │ Model (HR/Temp/SpO2)  │
                  │                           │               └───────────┬───────────┘
          ┌───────┴───────┐           ┌───────┴───────┐                   │
          ▼               ▼           ▼               ▼                   │
      ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐             │
      │ M1: FER   │ │ M2: Eye   │ │ Reflex    │ │ Dual SER  │             │
      │ Efficient │ │ MediaPipe │ │ Bypass    │ │ Wav2Vec2  │             │
      │ Net-B2    │ │ EAR 478pt │ │ Laughter/ │ │ + MFCC    │             │
      │ ONNX (~6ms│ │ PERCLOS   │ │ Crying    │ │ MLP ONNX  │             │
      └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘             │
            │             │             └───────┬─────┘                   │
            │             │                     │                         │
            ▼             ▼                     ▼                         ▼
      ┌───────────────────────────────────────────────────────────────────────┐
      │ M4: Quality-Gated Multimodal Fusion Engine (EMA alpha=0.15)           │
      │ Context Cross-Validation: HR High + Calm Affect = Physical Exertion   │
      └───────────────────────────────────┬───────────────────────────────────┘
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
      ┌───────────────────────┐ ┌───────────────────┐ ┌───────────────────────┐
      │ Fixed-Scale HUD Video │ │ SQLite Mission DB │ │ M7: Inactivity        │
      │ Overlay (30 FPS)      │ │ (maitri_logs.db)  │ │ Multi-Factor Watchdog │
      └───────────────────────┘ └───────────────────┘ └───────────┬───────────┘
                                                                  │ (Incapacity)
  ================================================================│=========================
  STATE GATEWAY (Decoupled, Read-Only Snapshot Interface at 1 Hz) │
  ================================================================│=========================
                  │                                               │
                  ▼                                               │
      ┌───────────────────────────────────────────────┐           │
      │ Structured State Snapshot                     │           │
      │ Stress, Trend, Confidence, Fatigue, Context   │           │
      └───────────────────────┬───────────────────────┘           │
                              ▼                                   │
      ┌───────────────────────────────────────────────┐           │
      │ Trend, Confidence & Persistence Filter        │           │
      │ Slope Evaluation, Dwell Time Thresholding     │           │
      └───────────────────────┬───────────────────────┘           │
                              │                                   │
  ============================│===================================│=========================
  PHASE 2: CLOSED-LOOP AFFECTIVE SUPPORT ENGINE (PROPOSED)        │
  ============================│===================================│=========================
                              ▼                                   │
      ┌───────────────────────────────────────────────┐           │
      │ Support Policy Engine: 6-State FSM            │           │
      │ STABLE -> MILD -> SUPPORT -> DE-ESC -> RECOV  │           │
      │ Governed by Hysteresis & Refractory Cooldown  │           │
      └───────┬───────────────────────────────┬───────┘           │
              │                               │                   │
   [Sub-Critical Concern]             [Severe Crisis]             │
              ▼                               │                   │
      ┌───────────────┐                       │                   │
      │ PRE-LLM GATE  │                       │                   │
      │ Risk Filter   │                       │                   │
      └───────┬───────┘                       │                   │
              │ (Normal Concern)              │                   │
              ▼                               │                   │
      ┌───────────────┐   ┌───────────────┐   │                   │
      │ Local KB      │   │ Adaptive      │   │                   │
      │ CBT/Grounding │   │ Audio Engine  │   │                   │
      └───────┬───────┘   │ Tempo Pacing  │   │                   │
              ▼           └───────┬───────┘   │                   │
      ┌───────────────┐           │           │                   │
      │ Local LLM     │           │           │                   │
      │ Quantized     │           │           │                   │
      │ Llama-3/Phi-3 │           │           │                   │
      └───────┬───────┘           │           │                   │
              ▼                   │           │                   │
      ┌───────────────┐           │           │                   │
      │ POST-LLM GATE │           │           │                   │
      │ Safety Check  │           │           │                   │
      └───────┬───────┘           │           │                   │
              │ (Pass)            │           │                   │
              ▼                   ▼           │                   │
      ┌───────────────┐   ┌───────────────┐   │                   │
      │ Offline Speech│   │ Ambient Cabin │   │                   │
      │ I/O (Whisper/ │   │ Audio Out     │   │                   │
      │ Piper TTS)    │   │ (Non-Critical)│   │                   │
      └───────┬───────┘   └───────┬───────┘   │                   │
              │                   │           │                   │
              └─────────┬─────────┘           │                   │
                        ▼                     │                   │
                 ┌─────────────┐              │                   │
                 │  ASTRONAUT  │              │                   │
                 └──────┬──────┘              │                   │
                        ▼                     │                   │
                 ┌─────────────┐              │                   │
                 │ Closed-Loop │              │                   │
                 │ Feedback    │              │                   │
                 │ (Delta Eval)│              │                   │
                 └──────┬──────┘              │                   │
                        │                     │                   │
           ┌────────────┴────────────┐        │                   │
           ▼                         ▼        ▼                   ▼
    [Improvement]              [Persistent / Critical] ────────────────────────┐
           │                                                                  ▼
           ▼                                                  ┌───────────────────────────────┐
   FSM -> RECOVERY                                            │ CRITICAL EMERGENCY PATH       │
                                                              │ Deterministic Alarms &        │
                                                              │ Delay-Tolerant Networking     │
                                                              │ (DTN Outbox to Earth Ground)  │
                                                              └───────────────────────────────┘
====================================================================================================
```

### Core Subsystem Modules:
1. **Facial Emotion Recognition (M1) [Implemented]:** Upgraded from VGG-Face to EfficientNet-B2 ONNX (`enet_b2_7.onnx`) with 6 threads, processing frames in ~6.3 ms. Enhanced with LAB-space CLAHE and adaptive gamma in `dip_enhancer.py`.
2. **Eye Tracking & Fatigue (M2) [Implemented]:** MediaPipe 478-point 3D FaceLandmarker computing EAR. Implements rising-edge blink debouncing over a 60-second window, PERCLOS fatigue tracking, and reading-stupor hysteresis (5 blinks/min threshold).
3. **Acoustic Processing & Dual SER (M_Voice) [Implemented]:** 75 Hz FFT high-pass filter in `dap_enhancer.py` eliminating cooling fan drone (ALC257 noise). Autocorrelation envelope reflex bypass for laughter/sobbing. Dual ensemble of Wav2Vec2 ONNX (86 MB) and synthetic-trained MFCC-MLP ONNX (<5 ms).
4. **Physiological Vitals Strain Model (M3) [Implemented]:** Piecewise homeostatic normalization of Heart Rate (50–160 BPM), Skin Temperature (35–40°C), and SpO2 (85–100%) with non-linear hypoxia penalties.
5. **Quality-Gated Multimodal Fusion Engine (M4) [Implemented]:** Dynamic quality gating ($W_i$), Exponential Moving Average temporal smoothing ($lpha=0.15$), and exercise cross-validation (capping stress during workout tachycardia).
6. **Inactivity Watchdog Subsystem (M5) [Implemented]:** Velocity tracking ($\ge 8$ px shift) and blink intervals detecting 45s immobility, triggering Check-In prompts and Incapacitation alerts.
7. **Mission Telemetry DB & Dossier Generator (M6) [Implemented]:** Thread-safe SQLite logging (`maitri_logs.db`) with dynamic schema migration and automated Markdown Flight Surgeon Dossier export.
8. **Delay-Tolerant Networking Simulator (M7) [Implemented]:** Simulates speed-of-light propagation delays (LEO, Moon, Mars) and queues priority-1 emergencies.
9. **State Gateway & Support Policy FSM (M8) [Proposed]:** Read-only 1 Hz JSON snapshot extractor; 6-state FSM (`STABLE` $	o$ `MILD` $	o$ `SUPPORT` $	o$ `DE-ESC` $	o$ `RECOV` $	o$ `REASSESS`) governed by dwell-time hysteresis and refractory cooldowns.
10. **Safety-Gated Conversational Engine (M9) [Proposed]:** Dual deterministic safety gates, local CBT/grounding knowledge base, quantized local LLM (Llama-3-8B / Phi-3), and offline speech I/O (`faster-whisper` + Piper TTS).
11. **Adaptive Audio & Feedback Engine (M10) [Proposed]:** Tempo-regulated instrumental audio (60–70 BPM), biophilic soundscapes, breathing pacing, and closed-loop delta computation ($\Delta 	ext{Stress}$).

### Techniques and Algorithms:
- **LAB-Space CLAHE & Adaptive Gamma:** Luminance histogram equalization and power-law gamma transformation correcting cabin lighting in `dip_enhancer.py`.
- **EfficientNet-B2 ONNX Classification:** Compound-scaled CNN executing in ~6.3 ms for 7-class facial affect classification.
- **Eye Aspect Ratio (EAR) Euclidean Geometry:** Eyelid aperture scalar computed from 6 Euclidean landmark distances per eye across a 478-point mesh.
- **Rising-Edge Blink Debouncing:** State-machine thresholding requiring EAR to dip below 0.25 and rise back above 0.25 to register a blink.
- **Reading-Stupor Hysteresis:** Context-aware lowering of active blink threshold from 10 to 5 blinks/min during manual reading tasks.
- **Infrasonic 75 Hz FFT High-Pass Filter:** Filters low-frequency life-support cooling fan drone (ALC257 fan rumble) in `dap_enhancer.py`.
- **Laughter / Crying Reflex Bypass:** Envelope autocorrelation periodicity ($R_{\text{env}} \sim 0.90$) forcing heuristic emotion overrides for laughter and sobbing.
- **Dual-Model SER Ensemble:** SNR-weighted combination of contextual Wav2Vec2 ONNX (86 MB) and synthetic-trained MFCC-MLP ONNX (<5 ms).
- **Piecewise Homeostatic Vitals Scaling:** Normalizes HR, Temp, and SpO2 into a standardized physiological strain index (0.0–1.0).
- **Dynamic Quality-Aware Weight Gating:** Re-normalizes modality weights based on real-time signal quality coefficients ($Q_i$).
- **Exponential Moving Average (EMA) Smoothing:** Temporal IIR filter ($lpha=0.15$) dampening high-frequency classification flicker.
- **Physical Exertion Cross-Validation:** Classifies workout tachycardia with calm facial affect as `Physical Exertion`, capping stress.
- **Multi-Factor Inactivity Watchdog:** Instantaneous bounding box velocity tracking ($\ge 8$ px shift) and blinks detecting 45s immobility.
- **Delay-Tolerant Networking (DTN) Queueing:** Asynchronous bundle storage simulator modeling speed-of-light propagation delays.
- **Finite State Machine (FSM) Policy Engine:** Deterministic 6-state model with dwell-time hysteresis (15s) and refractory cooldowns (180s).
- **Dual Deterministic Safety Gating:** Pre-LLM risk filter and Post-LLM output validator routing emergencies to DTN outbox.
- **Psychoacoustic Tempo-Regulated Audio:** Dynamic audio pacing (60–70 BPM) and biophilic soundscapes downregulating sympathetic arousal.
- **Closed-Loop Affective Delta Engine:** Mathematical computation of $\Delta 	ext{Stress}$ paired with discrete astronaut self-reports (`Better`, `Same`, `Worse`).

## 6.2 Software and Hardware Requirements and Availability

### Software Requirements:
- **Operating System:** Linux (Ubuntu 22.04 LTS / Fedora 39) or Windows 11 with WSL2.
- **Programming Language:** Python 3.11.x (pinned in local virtual environments `venv` and `venv311`).
- **Inference & Perception:** `onnxruntime` ($\ge 1.16.0$), `mediapipe` ($\ge 0.10.9$), `opencv-python-headless` ($\ge 4.8.0$), `numpy` ($\ge 1.24.0$), `scipy` ($\ge 1.11.0$).
- **Audio & Speech:** `librosa` ($\ge 0.10.1$), `sounddevice` ($\ge 0.4.6$), `PyAudio`, `faster-whisper` (Phase 2), `piper-tts` (Phase 2).
- **Application Framework:** `streamlit` ($\ge 1.30.0$), `streamlit-webrtc` ($\ge 0.47.0$ with `aiortc`), native Python `threading` with mutex locks.
- **Database & Telemetry:** `sqlite3` (embedded database `maitri_logs.db`).

### Hardware Requirements:
- **Development & Benchmarking System:** Multi-core x86_64 CPU (Intel Core i7-11800H / AMD Ryzen 7 5800H, 8 Cores, 16 Threads), 16 GB RAM, 512 GB NVMe SSD. Optional dedicated GPU (NVIDIA RTX 3060 / 4060 with 6GB+ VRAM) with pure multi-threaded CPU fallback.
- **Sensory Hardware:** Standard USB UVC HD Webcam (720p/1080p @ 30 FPS), Cardioid USB Microphone (16 kHz / 44.1 kHz), and simulated physiological biosensor stream.
- **Simulated Flight Client:** Standard crew laptop (e.g., ISS Lenovo ThinkPad T-series or HP ZBook).

### Availability:
- All required software tools are freely available under permissive open-source licenses (Apache 2.0, MIT, BSD) with zero ongoing cloud costs.
- Standard hardware components are readily available in consumer and industrial aerospace test environments.
- 100% local, zero-cloud execution ensures long-term operational sustainability across multi-year deep-space missions.

---

\newpage

# 7 Expected Outcome

After successful implementation, the MAITRI system will provide the following outcomes:

## Verified Phase 1 Outcomes:
- Continuous zero-lag 30 FPS live video pipeline with sub-10ms decoupled inference (~6.3 ms for FER).
- Dynamic quality-gated mathematical fusion producing a stable Stress Index (0–100%) with EMA temporal smoothing ($\alpha=0.15$).
- Elimination of false vocal classifications caused by life-support cooling fan drone via 75 Hz infrasonic high-pass filtering and heuristic reflex overrides.
- Accurate differentiation of physical workout tachycardia from acute panic, and quiet manual reading from catatonia.
- Operational embedded SQLite logging, multi-factor inactivity watchdog, automated Flight Surgeon Dossiers, and simulated DTN deep-space queuing.

## Projected Phase 2 Outcomes:
- Completely offline support architecture running asynchronously without degrading the 30 FPS monitoring pipeline.
- Deterministic 6-state FSM eliminating intervention flapping through 15s dwell-time persistence, 10% hysteresis, and 180s cooldowns.
- Local quantized LLM providing empathetic conversational dialogue strictly bounded by dual deterministic safety gates.
- Tempo-regulated instrumental and biophilic acoustic pacing automatically suppressed during critical flight operations.
- Quantitative post-intervention evaluation ($\Delta \text{Stress}$) and user feedback driving iterative adaptation or ground escalation.

---

\newpage

# 8 Schedule

- Literature Survey & Problem Definition – Weeks 1–2 [COMPLETED]
- Baseline WebRTC Pipeline & DeepFace Testing – Weeks 2–3 [COMPLETED]
- SOTA FER Upgrade (EfficientNet-B2 ONNX) & Decoupling – Weeks 3–4 [COMPLETED]
- Eye Dynamics & PERCLOS Fatigue Modeling – Weeks 4–5 [COMPLETED]
- Audio Pipeline & DAP 75 Hz Filter Design – Weeks 5–6 [COMPLETED]
- Dual-Model SER & Vitals Normalization – Weeks 6–7 [COMPLETED]
- Quality-Aware Multimodal Fusion & Exercise Rules – Weeks 7–8 [COMPLETED]
- Aerospace Watchdog, DTN Outbox & Dossier Generator – Weeks 8–9 [COMPLETED]
- Phase 2 State Gateway & 6-State FSM Design – Weeks 9–10 [PROPOSED]
- Local Knowledge Base & Safety Gate Engineering – Weeks 10–11 [PROPOSED]
- Local LLM Benchmarking & Offline Speech I/O – Weeks 11–12 [PROPOSED]
- Adaptive Audio & Closed-Loop Feedback Delta Engine – Weeks 12–13 [PROPOSED]
- End-to-End System Integration & Stress Testing – Week 13 [PROPOSED]
- Documentation, Demonstration & Final Defense – Week 14 [PROPOSED]

---

\newpage

# Expected Challenges, Risks, and Mitigation Strategies

| Risk ID | Category | Potential Risk / Challenge | Severity | Operational Impact | Proposed Mitigation Strategy |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **R1** | **Acoustic Signal Processing** | Spacecraft Environmental Noise (life-support fans, pumps, ECLSS) corrupting speech signals and causing classification drift. | High | Microphone picks up constant background drone, biasing neural networks toward false "sad" or "angry" classifications. | Implement the 75 Hz FFT high-pass filter to attenuate low-frequency fan hum; enforce energy-based voice activity detection (VAD) that zeroes out acoustic weight during silence. |
| **R2** | **Computer Vision & Hardware** | Fluctuating Cabin Illumination and Rapid Head Motion causing facial landmark loss or reticle jitter. | Medium | Extreme glare or low light degrades facial cropping; optical flow tracking reticle experiences erratic scaling. | Preprocess frames with LAB-space CLAHE and adaptive gamma; utilize MediaPipe SSD isotropic square cropping; lock HUD text to fixed 480p coordinates; switch weight to vitals when face is lost. |
| **R3** | **Computational Feasibility** | Concurrency Contention and Latency Jitter between real-time Phase 1 video processing and Phase 2 local LLM execution. | High | Running a 7B/8B parameter LLM on CPU/GPU can starve the WebRTC video thread, dropping FPS and causing UI freezing. | Fully decouple Phase 2 from Phase 1 via an asynchronous background thread; pass lightweight 1 Hz JSON snapshots instead of video frames; utilize 4-bit quantized models (GGUF / ONNX INT8). |
| **R4** | **Operational Human Factors** | Intervention Flapping and Alarm Fatigue caused by noisy sensor readings oscillating around threshold boundaries. | High | Repeatedly triggering and canceling support prompts ("Intervention ON / Intervention OFF") irritates astronauts, leading to system shutdown. | Enforce an FSM with a 15-second dwell-time persistence filter, a 10% hysteresis band between activation and deactivation, and an enforced 180-second post-intervention refractory cooldown. |
| **R5** | **Clinical Human Factors** | Misclassifying Healthy Operational Exertion (e.g., daily 2.5-hour treadmill/CEVIS workouts) as acute psychological panic. | Medium | Elevated heart rate and perspiration during exercise trigger inappropriate psychiatric grounding alerts. | Program clinical cross-validation rules: if tachycardia occurs in the presence of calm facial expressions and active movement, categorize state as `Physical Exertion` and cap the Stress Index. |
| **R6** | **AI Safety & Alignment** | LLM Hallucination, Inappropriate Medical Advice, or Conversational Escalation during acute astronaut distress. | Critical | An unconstrained LLM generates medically invalid guidance, fails to recognize a life-threatening crisis, or exacerbates astronaut anxiety. | Restrict the LLM strictly to conversational reframing; enforce dual deterministic safety gates (Pre-LLM risk filter, Post-LLM output validator); route all critical emergencies directly to cabin alarms and the DTN outbox. |
| **R7** | **Communication & Network** | Complete Communication Blackout (e.g., Mars solar conjunction) preventing ground synchronization. | Medium | Terrestrial flight surgeons remain unaware of astronaut status during prolonged solar occultations. | Maintain an embedded local SQLite telemetry log (`maitri_logs.db`); bundle critical medical summaries into the DTN outbox, which automatically dispatches queued packets upon link restoration. |

---

\newpage

# References

1. Basner, M., Dinges, D. F., Mollicone, D., Ecker, A., Jones, C. W., Hyder, E. C., Di Antonio, A., Savelev, I., Ruf, G. G., & Sutton, J. P. (2014). "Psychological and behavioral changes during confinement in a 520-day simulated Mars mission to determine adaptation to spaceflight." *PLoS ONE*, 9(3), e93298. https://doi.org/10.1371/journal.pone.0093298
2. Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." *Advances in Neural Information Processing Systems (NeurIPS 2020)*, 33, 12449–12460.
3. Beck, J. S. (2011). *Cognitive Behavior Therapy: Basics and Beyond* (2nd ed.). The Guilford Press, New York.
4. Cerf, V., Burleigh, S., Hooke, A., Torgerson, L., Durst, R., Scott, K., Fall, K., & Weiss, H. (2007). "Delay-Tolerant Networking Architecture." *Internet Engineering Task Force (IETF) RFC 4838*. https://doi.org/10.17487/RFC4838
5. Chlan, L. L., Weinert, C. R., Heiderscheit, A., Tracy, M. F., Skaar, D. J., Guttormson, J. L., & Savik, K. (2013). "Effects of patient-directed music therapy on anxiety and sedative exposure in critically ill patients: a randomized clinical trial." *JAMA*, 309(22), 2335–2344. https://doi.org/10.1001/jama.2013.5670
6. Dinges, D. F., Venkataraman, S., McGlinchey, E. L., & Metaxas, D. N. (2007). "Optical Computer Recognition of Stress, Affect and Fatigue." *NASA Technical Reports Server (NTRS)*, NASA/CR-2007-214811.
7. Ekman, P., & Friesen, W. V. (1978). *Facial Action Coding System: A Technique for the Measurement of Facial Movement*. Consulting Psychologists Press, Palo Alto, CA.
8. Kanas, N., Salnitskiy, V., Grund, E. M., Weiss, D. S., Gushin, V., Bostrom, A., Stewart, C., Kozerenko, O., & Marmar, C. R. (2001). "Psychosocial issues in space: results from spaceflights." *Journal of Psychosomatic Research*, 51(6), 727–734. https://doi.org/10.1016/S0022-3999(01)00244-6
9. Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C. L., Yong, M. G., Lee, J., Chang, W. T., Hua, W., Georg, M., & Grundmann, M. (2019). "MediaPipe: A Framework for Building Perception Pipelines." *arXiv preprint arXiv:1906.08172*.
10. NASA Human Research Program. (2021). "Risk of Adverse Cognitive or Behavioral Conditions and Psychiatric Disorders." *Human Research Roadmap (HRR)*, NASA Johnson Space Center, Houston, TX. Available at: https://humanresearchroadmap.nasa.gov/
11. Picard, R. W. (1997). *Affective Computing*. MIT Press, Cambridge, MA. https://doi.org/10.7551/mitpress/1140.001.0001
12. Soukupova, T., & Cech, J. (2016). "Real-Time Eye Blink Detection using Facial Landmarks." *Proceedings of the 21st Computer Vision Winter Workshop (CVWW 2016)*, Rimske Toplice, Slovenia.
13. Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *Proceedings of the 36th International Conference on Machine Learning (ICML 2019)*, PMLR 97:6105–6114.
14. Thayer, J. F., & Lane, R. D. (2000). "A model of neurovisceral integration in emotion regulation and dysregulation." *Journal of Affective Disorders*, 61(3), 201–216. https://doi.org/10.1016/S0165-0327(00)00338-4
15. Toups, P. O., et al. (2018). "Autonomous Medical Decision Support: Evolution of Medical Operations for Deep Space Exploration." *NASA Technical Memorandum*, NASA/TM-2018-220072, NASA Johnson Space Center.

---

\newpage

# Details of Team Members

| Roll No | Name of the Student | Contact No. | Email ID | Signature |
| :---: | :--- | :---: | :--- | :---: |
| **23031012** | **Mr. MUKUL RAMESH KUMBHAR** | +91 98344 01315 | mukulkumbhar2005@gmail.com | _____________ |
| **23031016** | **Mr. ADITYA ANANDA KUMBHAR** | +91 87660 46514 | kumbharaditya777@gmail.com | _____________ |
| **23101058** | **Mr. ARYAN AMARSINH PATIL** | +91 91725 34221 | aryanpatil.iot@gmail.com | _____________ |
| **23041084** | **Mr. SHREEDHAR SURESH DALVI** | +91 94211 88392 | shreedhardalvi@gmail.com | _____________ |
| **23101007** | **Ms. PRANJAL SAMPATRAO MANE** | +91 98233 45110 | pranjalmane.iot@gmail.com | _____________ |

<br>

**Date:** 25th September 2026  
**Place:** ADCET, Ashta (Sangli), Maharashtra, India

<br><br>

```text
       _________________________________                _________________________________
             Dr. SAKTHIPRIYA BALU                              Dr. AMOL S. DANGE
                    GUIDE                                  Capstone Project Coordinator
       Dept. of Aeronautical Engineering                 Dept. of Aeronautical Engineering
```

<br><br><br>

```text
       _________________________________                _________________________________
              Head of Department                                    Director
       Dept. of Aeronautical Engineering                          ADCET, Ashta
          ADCET, Ashta (Kolhapur)                         (Autonomous Institute)
```

---

\newpage

# Annexure - I: ADCET Capstone Project Proposal (Proforma - I)

| S.NO. | PARTICULARS | DETAILS |
| :---: | :--- | :--- |
| **1** | **Student Names with URN** | 1. Mukul Ramesh Kumbhar (URN: 23031012)<br>2. Aditya Ananda Kumbhar (URN: 23031016)<br>3. Aryan Amarsinh Patil (URN: 23101058)<br>4. Shreedhar Suresh Dalvi (URN: 23041084)<br>5. Pranjal Sampatrao Mane (URN: 23101007) |
| **2** | **Project Title** | **MAITRI: Multimodal AI-based Astronaut Telemetry & Real-time Intelligence** *(Multimodal Astronaut Psychological Well-being and Emotional Support System)* |
| **3** | **Objective** | **Phase 1 (Implemented Foundation):** Develop a 100% offline, zero-lag, multimodal monitoring pipeline combining real-time Facial Emotion Recognition (FER), Eye Aspect Ratio (EAR) blink-fatigue tracking, Speech Emotion Recognition (SER), and physiological vitals normalization into a unified Stress Index ($0-100\%$) backed by a mission watchdog and Delay-Tolerant Networking (DTN) simulator.<br>**Phase 2 (Proposed Development):** Formulate and integrate a safety-gated, closed-loop, offline affective support architecture operating on the paradigm *Sense $\to$ Assess $\to$ Select $\to$ Support $\to$ Reassess $\to$ Adapt*, utilizing a deterministic Finite State Machine (FSM), local conversational LLM, offline speech I/O, scientifically grounded adaptive audio, and objective/subjective feedback adaptation. |
| **4** | **Problem Statement** | In Long-Duration Spaceflight (LDSF), extreme confinement, sleep disruption, and light-speed communication delays (3 to 22 minutes roundtrip for Mars) render Earth-based psychiatric tele-support impossible. Subjective self-reporting is vulnerable to operational suppression, while single-modality sensors fail to disambiguate physical exertion from psychological panic. There is an urgent need for an onboard, edge-native, privacy-preserving AI system that autonomously detects mental-physical degradation and administers closed-loop non-pharmacological support without cloud dependence. |
| **5** | **Scope of the Project** | **Included:** Non-invasive computer vision (FER, EAR, blink rate), acoustic prosody analysis (dual-model SER), physiological vitals strain modeling (HR, Temp, SpO2), quality-gated multimodal fusion, inactivity/catatonia watchdog, local SQLite telemetry logging, DTN ground outbox simulator, deterministic FSM support policy, local knowledge base, quantized local LLM conversational support, offline STT/TTS, adaptive audio pacing, and closed-loop feedback adaptation.<br>**Excluded / Limitations:** Not an autonomous clinical psychiatric authority; does not provide formal medical diagnoses or pharmacological treatment; does not replace human flight surgeons; evaluated on simulated hardware testbeds rather than certified spaceflight avionics. |
| **6** | **Expected Outcomes** | 1. Sub-10ms decoupled visual and acoustic inference pipeline executing at 30 FPS locally.<br>2. Dynamic quality-gated stress estimation robust to camera occlusion, ambient fan noise, and exercise.<br>3. Deterministic safety-gated FSM support policy preventing intervention flapping.<br>4. Closed-loop adaptation verifying post-intervention stress reduction via biosignal and user feedback.<br>5. Comprehensive mission dossier generation and delayed telemetry queueing for ground control. |
| **7** | **Timeline** | 14-week phased execution across Academic Year 2025–2026. Weeks 1–8: Literature survey, Phase 1 multimodal monitoring architecture, optimization, and testing (COMPLETED). Weeks 9–14: Phase 2 State Gateway, FSM policy engine, local LLM/audio integration, closed-loop feedback, end-to-end testing, and capstone documentation (CURRENT & PROPOSED). |
| **8** | **Methodology** | **Paradigm:** *Sense $\to$ Assess $\to$ Select $\to$ Support $\to$ Reassess $\to$ Adapt*.<br>**Visual Pipeline:** MediaPipe SSD + Digital Image Processing (CLAHE, Gamma) + EfficientNet-B2 ONNX + 478-landmark Eye Aspect Ratio (EAR) with rising-edge blink debouncing.<br>**Acoustic Pipeline:** Infrasonic 75 Hz high-pass filter + Laughter/Crying reflex bypass + Dual-Model SER (Wav2Vec2 ONNX + MFCC-MLP ONNX).<br>**Physiological Modeling:** Piecewise normalization of HR, Temp, and SpO2.<br>**Fusion:** Quality-gated Exponential Moving Average (EMA, $\alpha=0.15$) with physical exertion cross-validation.<br>**Decision & Support:** Multi-Factor Watchdog + State Gateway Snapshot + 6-State FSM with dwell/cooldown hysteresis + Dual Pre/Post Safety Gates + Local Knowledge Base (CBT/Grounding) + Quantized Local LLM + Offline STT/TTS + Adaptive Audio + Closed-Loop Feedback Delta Engine. |
| **9** | **Resources Required** | **Hardware:** Multi-core workstation/laptop (AMD Ryzen / Intel Core i7, 16 GB RAM), USB HD Webcam (1080p@30FPS), Cardioid USB Microphone, Wearable Biosensor simulator.<br>**Software / Libraries:** Python 3.11, ONNX Runtime, MediaPipe Tasks, OpenCV, Streamlit & Streamlit-WebRTC (aiortc), NumPy, SciPy, Librosa, SQLite3, faster-whisper, Piper TTS, quantized LLM execution engine (llama.cpp / Ollama).<br>**Availability:** 100% open-source, locally hosted, zero cloud costs. |
| **10** | **Expected Challenges and Risks** | 1. *Acoustic Ambient Interference:* Spacecraft cabin life-support / cooling fan noise causing false vocal affect predictions (Mitigated by 75 Hz infrasonic filtering and silence energy gating).<br>2. *Inference Latency on Edge Compute:* Multi-model concurrency inducing frame drops (Mitigated by ONNX INT8 quantization and asynchronous worker thread decoupling).<br>3. *Intervention Flapping / Alarm Fatigue:* Rapid oscillation between stress states (Mitigated by FSM dwell-time hysteresis and refractory cooldown timers).<br>4. *Hallucination / Inappropriate LLM Generation:* Uncontrolled conversational outputs during high stress (Mitigated by dual deterministic safety gates and retrieval-grounded local knowledge bases). |
