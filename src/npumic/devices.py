"""Microphone discovery and the system's default input through PipeWire."""

import json
import subprocess
from dataclasses import dataclass

VIRTUAL_MIC_PREFIX = "npumic"


@dataclass
class Mic:
    name: str  # PipeWire node.name; stable across reconnects
    label: str
    bluetooth: bool
    is_default: bool
    muted: bool = False  # muted in system sound settings; delivers silence
    usb: bool = False


def _pw_dump() -> list[dict]:
    return json.loads(subprocess.run(["pw-dump"], capture_output=True, text=True, check=True).stdout)


@dataclass
class Output:
    name: str  # PipeWire node.name
    label: str


def _default(objects: list[dict], key: str) -> str | None:
    for obj in objects:
        if obj.get("type") == "PipeWire:Interface:Metadata" and obj.get("props", {}).get("metadata.name") == "default":
            for entry in obj.get("metadata", []):
                if entry.get("key") == key:
                    return (entry.get("value") or {}).get("name")
    return None


def _label(props: dict) -> str:
    return props.get("node.description") or props.get("node.nick") or props.get("node.name", "")


def _node_id(objects: list[dict], name: str) -> int | None:
    for obj in objects:
        props = (obj.get("info") or {}).get("props") or {}
        if obj.get("type") == "PipeWire:Interface:Node" and props.get("node.name") == name:
            return obj["id"]
    return None


def node_id(name: str) -> int | None:
    return _node_id(_pw_dump(), name)


def default_output() -> Output | None:
    """The system's current default playback device (headphones, speakers)."""
    objects = _pw_dump()
    name = _default(objects, "default.audio.sink")
    for obj in objects:
        props = (obj.get("info") or {}).get("props") or {}
        if name and props.get("node.name") == name:
            return Output(name=name, label=_label(props))
    return None


def default_source() -> str | None:
    """Node name of the system's current default input."""
    return _default(_pw_dump(), "default.audio.source")


def set_default_source(name: str) -> bool:
    """Makes the node `name` the system's default input; False if it does not exist.

    Writes WirePlumber's configured default directly: `wpctl set-default`
    refuses virtual sources such as this app's microphone.
    """
    if _node_id(_pw_dump(), name) is None:
        return False
    subprocess.run(["pw-metadata", "0", "default.configured.audio.source", json.dumps({"name": name}), "Spa:String:JSON"],
                   capture_output=True, check=True)
    return True


def list_mics() -> list[Mic]:
    """Physical microphones currently connected, excluding this app's own virtual mic."""
    objects = _pw_dump()
    default = _default(objects, "default.audio.source")
    mics = []
    for obj in objects:
        props = (obj.get("info") or {}).get("props") or {}
        name = props.get("node.name", "")
        if props.get("media.class") != "Audio/Source" or name.startswith(VIRTUAL_MIC_PREFIX):
            continue
        mics.append(
            Mic(
                name=name,
                label=_label(props),
                bluetooth=props.get("device.api") == "bluez5",
                is_default=name == default,
                muted=any(p.get("mute") for p in ((obj.get("info") or {}).get("params") or {}).get("Props", [])),
                usb=props.get("device.bus") == "usb",
            )
        )
    return mics


def pick_mic(mics: list[Mic], preferred: str | None = None) -> Mic | None:
    """The microphone to use: `preferred` when connected, else USB, then Bluetooth, then built-in.

    Muted microphones come last.
    """
    for mic in mics:
        if mic.name == preferred:
            return mic

    def rank(mic: Mic) -> tuple[bool, int]:
        return mic.muted, 0 if mic.usb else 1 if mic.bluetooth else 2

    return min(mics, key=rank, default=None)
