import time
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class WatchdogStatus:
    state: str
    alert_message: str

class NativeWatchdog:
    def __init__(self, timeout_sec: float = 45.0):
        self.timeout_sec = timeout_sec
        self.last_activity_time = time.time()
        self.state = "NORMAL" # NORMAL, CHECK_IN, INCAPACITATED
        
    def trigger_simulation(self):
        """Forces the watchdog into CHECK_IN state for presentation purposes."""
        self.state = "CHECK_IN"
        self.last_activity_time = time.time() - self.timeout_sec
        
    def acknowledge(self):
        """Manual override from the dashboard."""
        self.state = "NORMAL"
        self.last_activity_time = time.time()
        
    def update(self, is_speaking: bool, ear: float, face_quality: float) -> WatchdogStatus:
        now = time.time()
        
        # If the user is speaking, has eyes open, or face is clearly visible, they are active.
        if is_speaking or ear > 0.22 or face_quality > 0.5:
            if self.state == "NORMAL":
                self.last_activity_time = now
                
        time_since_active = now - self.last_activity_time
        
        if self.state == "NORMAL":
            if time_since_active >= self.timeout_sec:
                self.state = "CHECK_IN"
                
        elif self.state == "CHECK_IN":
            if time_since_active >= self.timeout_sec + 15.0:
                self.state = "INCAPACITATED"
                
        # Generate alert message
        if self.state == "NORMAL":
            return WatchdogStatus("NORMAL", "")
        elif self.state == "CHECK_IN":
            countdown = max(0, int((self.timeout_sec + 15.0) - time_since_active))
            return WatchdogStatus("CHECK_IN", f"ASTRONAUT INACTIVITY DETECTED. CHECK-IN REQUIRED WITHIN {countdown}s.")
        else:
            return WatchdogStatus("INCAPACITATED", "SYSTEM INCAPACITATION PROTOCOL ENGAGED.")

@dataclass
class DTNPacket:
    packet_id: str
    priority: str
    summary: str
    status: str

class NativeDTNOutbox:
    def __init__(self):
        self.packets: List[DTNPacket] = []
        self.latency_sec = 0.5
        self.mission_name = "ISRO Gaganyaan (LEO)"
        
    def set_mission(self, name: str, latency: float):
        self.mission_name = name
        self.latency_sec = latency
        
    def format_latency(self) -> str:
        if self.latency_sec < 1.0:
            return f"{int(self.latency_sec * 1000)} ms"
        else:
            return f"{self.latency_sec:.1f} s"
            
    def dispatch(self, pkt_id: str, priority: str, summary: str):
        self.packets.insert(0, DTNPacket(pkt_id, priority, summary, "QUEUED (WAITING FOR GROUND LINK)"))

def compile_native_dossier(crew: str, mission: str, hrm: float, spo2: float, stress: float) -> str:
    """Generates a simple Markdown clinical dossier."""
    return f"""# 🛰️ FLIGHT SURGEON CLINICAL DOSSIER
**CREW MEMBER:** {crew}  
**MISSION PROFILE:** {mission}  
**TIMESTAMP (MET):** T+ 00:00:00

## 1. PHYSIOLOGICAL VITALS
- **Heart Rate:** {hrm} BPM
- **SpO2:** {spo2}%

## 2. PSYCHOLOGICAL ASSESSMENT
- **Stress Strain Index:** {stress:.1f}%

## 3. CLINICAL SUMMARY
Automated telemetry audit compiled by MAITRI 2.0 system.
"""
