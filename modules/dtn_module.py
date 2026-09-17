"""
MAITRI 2.0 — M8: Delay-Tolerant Networking (DTN) Ground Telemetry Outbox
Simulates NASA / CCSDS Bundle Protocol (RFC 5050) deep-space communication link
with dynamic speed-of-light propagation latency based on the active mission profile.

Bandwidth Conservation Protocol:
- Routine telemetry is logged exclusively to the local spacecraft database.
- The Ground DTN Outbox is strictly reserved for Priority-1 Emergencies
  (Incapacitation, Hypoxia Desaturation, Critical Stress) and Flight Surgeon Audits.
"""

import time
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DTNPacket:
    packet_id: str
    priority: str                     # "CRITICAL-1 (EMERGENCY)" / "CLINICAL-AUDIT"
    packet_type: str                  # "INCAPACITATION_INCIDENT" / "HYPOXIA_EMERGENCY" / "CLINICAL_AUDIT_DOSSIER"
    mission_profile: str              # "ISRO Gaganyaan (LEO)" / "Mars Transit", etc.
    timestamp_utc: str
    met_timestamp: str                # Mission Elapsed Time string
    latency_sec: float                # One-way propagation delay in seconds
    status: str                       # "TRANSMITTING" / "DELIVERED TO GROUND"
    payload_summary: str
    details: Dict[str, Any] = field(default_factory=dict)


class DTNOutbox:
    """
    Onboard Delay-Tolerant Network Telemetry Buffer.
    Ensures critical telemetry survives communication blackouts, occultation, and planetary distances.
    Only transmits high-priority emergency packets to preserve deep-space RF bandwidth.
    """

    def __init__(self, ground_delay_seconds: float = 0.5, mission_name: str = "ISRO Gaganyaan (LEO)"):
        self.ground_delay = ground_delay_seconds
        self.mission_name = mission_name
        self.packet_counter = 101
        self.packets: List[DTNPacket] = []

    def set_mission_profile(self, mission_name: str, latency_sec: float) -> None:
        """Update mission profile and propagation delay."""
        self.mission_name = mission_name
        self.ground_delay = latency_sec

    def format_latency_str(self) -> str:
        """Formats the current latency into an aerospace human-readable string."""
        if self.ground_delay < 1.0:
            return f"{self.ground_delay:.1f}s (Direct S-Band Link)"
        elif self.ground_delay < 60.0:
            return f"{self.ground_delay:.1f}s (Relay Satellite)"
        else:
            mins = int(self.ground_delay // 60)
            secs = int(self.ground_delay % 60)
            return f"{mins}m {secs:02d}s (DTN Active)"

    def dispatch_incident_packet(
        self,
        incident_type: str,
        met_str: str,
        summary: str,
        details: Dict[str, Any],
        priority: str = "CRITICAL-1 (EMERGENCY)",
    ) -> DTNPacket:
        """Enqueues a high-priority emergency medical incident packet for ground transmission."""
        self.packet_counter += 1
        pkt_id = f"PKT-{self.packet_counter:04d}"
        now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        status_text = (
            f"TRANSMITTING TO GROUND (ETA: +{self.format_latency_str()})"
            if self.ground_delay > 1.0
            else "DIRECT TELEMETRY UPLINK CONFIRMED"
        )

        pkt = DTNPacket(
            packet_id=pkt_id,
            priority=priority,
            packet_type=incident_type,
            mission_profile=self.mission_name,
            timestamp_utc=now_str,
            met_timestamp=met_str,
            latency_sec=self.ground_delay,
            status=status_text,
            payload_summary=summary,
            details=details,
        )
        self.packets.insert(0, pkt)  # newest first
        return pkt

    def get_packets(self) -> List[DTNPacket]:
        return self.packets
