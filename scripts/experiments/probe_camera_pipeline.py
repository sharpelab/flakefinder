"""Probe A: dump the K5C's implemented UCAPI property space.

Read-only camera probe for the camera-raw investigation (2026-08-06).
Connects to the SDK, initializes the camera unit, and for every UCAPI
property ID in 16384-16505 reports: implemented?, value type, current
value, and (via PropertyInfo reflection) enum options / range min-max-step.

Key questions this answers:
  - PROP_PIXEL_TYPE options: is raw Bayer / mono delivery exposed?
  - PROP_PIXEL_DEPTH options: is 12-bit exposed?
  - PROP_COLOUR_TEMPERATURE options: which index is UserDefinedMatrix?
  - GAIN_RED/GREEN/BLUE ranges: confirms the min-1.0 clamp, reveals the cap.
  - SHARPENING_ENABLED actual state under our stack.

Hardware footprint: opens the SDK connection and calls camera unit Init()
(required before properties are readable; resets camera settings to SDK
defaults, same as every normal capture session). NO acquisition, NO stage,
NO Z, NO lamp, NO shutter.

Usage (on the microscope PC):
    uv run python scripts/experiments/probe_camera_pipeline.py [--json out.json]
"""

import argparse
import contextlib
import json

from flakefinder.leica.core import LeicaConnection, get_interface
from flakefinder.leica.enums import IID, UCAPI_TID

# Full UCAPI 2023.3 property ID -> name map (ucapi.h enum order from
# __UCAPI_PROPID_START = 16384; validated against known IDs).
PROP_NAMES = {
    16384: "SERIAL_NUMBER",
    16385: "FIRMWARE_HEAD_VERSION",
    16386: "VERSION_INFO",
    16387: "SENSOR_XRESOLUTION",
    16388: "SENSOR_YRESOLUTION",
    16389: "PHYSICAL_PIXEL_XSIZE",
    16390: "PHYSICAL_PIXEL_YSIZE",
    16391: "LOGICAL_XRESOLUTION",
    16392: "LOGICAL_YRESOLUTION",
    16393: "LOGICAL_CENTRE_XOFFSET",
    16394: "LOGICAL_CENTRE_YOFFSET",
    16395: "LOGICAL_PIXEL_XSIZE",
    16396: "LOGICAL_PIXEL_YSIZE",
    16397: "IMAGE_TRIGGER_MODE",
    16398: "TRIGGER_POLARITY",
    16399: "EXPOSURE_TIME",
    16400: "EMGAIN",
    16401: "GAIN",
    16402: "GAIN_RED",
    16403: "GAIN_GREEN",
    16404: "GAIN_BLUE",
    16405: "AUTO_BRIGHTNESS_ENABLED",
    16406: "AUTO_BRIGHTNESS_LEVEL",
    16407: "AUTO_BRIGHTNESS_ROI",
    16408: "AUTO_BRIGHTNESS_ROI_ENABLED",
    16409: "COLOUR_SATURATION",
    16410: "BINNING_LEVEL",
    16411: "BINNED_BRIGHTNESS_CORRECTION_ENABLED",
    16412: "READOUT_SPEED",
    16413: "PIXEL_DEPTH",
    16414: "PIXEL_TYPE",
    16415: "READOUT_REGION",
    16416: "READOUT_REGIONS",
    16417: "GAMMA_LEVEL",
    16418: "BLACK_LEVEL",
    16419: "WHITE_LEVEL",
    16420: "AUTO_GAMMA_LEVEL_ENABLED",
    16421: "AUTO_BLACK_LEVEL_ENABLED",
    16422: "AUTO_WHITE_LEVEL_ENABLED",
    16423: "SHARPENING_LEVEL",
    16424: "XMIRRORING_ENABLED",
    16425: "YMIRRORING_ENABLED",
    16426: "IMAGE_AVERAGING_MODE",
    16427: "IMAGE_AVERAGING_COUNT",
    16428: "CAPTURE_FORMAT_ID",
    16429: "STREAMING_FORMAT_ID",
    16430: "ACQUISITION_MODE",
    16431: "FAN_SPEED_LEVEL",
    16432: "CONTINUOUS_WHITE_BALANCE_ENABLED",
    16433: "CONTINUOUS_WHITE_BALANCE_ROI",
    16434: "CONTINUOUS_WHITE_BALANCE_ROI_ENABLED",
    16435: "FOCUS_SCORE_ROI",
    16436: "FOCUS_SCORE_ROI_ENABLED",
    16437: "WHITE_SHADING_REFERENCE_FILE",
    16438: "WHITE_SHADING_CORRECTION_ENABLED",
    16439: "SATURATION_CORRECTION",
    16440: "HUE_CORRECTION",
    16441: "AVERAGE_COLOUR_ROI",
    16442: "AVERAGE_COLOUR_ROI_ENABLED",
    16443: "PSEUDO_COLOUR_WAVELENGTH",
    16444: "BLACK_BALANCE_LEVEL",
    16445: "BLACK_BALANCE_ENABLED",
    16446: "ACQUISITION_TIMEOUT",
    16447: "AUTO_ACQUISITION_TIMEOUT_ENABLED",
    16448: "TEMPERATURE",
    16449: "TARGET_TEMPERATURE",
    16450: "FAN_SPEED",
    16451: "MIN_TRIGGER_INTERVAL",
    16452: "IMAGE_READOUT_TIME",
    16453: "NIR_MODE_ENABLED",
    16454: "CAMERA_SDK_VERSION",
    16455: "READOUT_OVERLAP_ENABLED",
    16456: "EMCCD_MODE",
    16457: "HOTPIXEL_EXPOSURE_THRESHOLD",
    16458: "HOTPIXEL_INTENSITY_THRESHOLD",
    16459: "MICROSCANNING_MODE",
    16460: "VERTICAL_SHIFT_TIME",
    16461: "VERTICAL_SHIFT_VOLTAGE_OFFSET",
    16462: "COLOUR_TEMPERATURE",
    16463: "NOISE_REDUCTION_LEVEL",
    16464: "NOISE_REDUCTION_ENABLED",
    16465: "MICRO_SCANNING_SHARPENING_LEVEL",
    16466: "MICRO_SCANNING_SHARPENING_ENABLED",
    16467: "BLACK_LOCK_THRESHOLD",
    16468: "BLACK_LOCK_ENABLED",
    16469: "WHITE_LOCK_THRESHOLD",
    16470: "WHITE_LOCK_ENABLED",
    16471: "ELECTRONCOUNT_SCALING",
    16472: "EMCCD_NOISE_REDUCTION_THRESHOLD",
    16473: "EMCCD_NOISE_REDUCTION_ENABLED",
    16474: "ENHANCED_ACQUISITION_MODE",
    16475: "HDR_MODE",
    16476: "HDR_CONTRAST",
    16477: "TEMPERATURE_PELTIER_ENABLED",
    16478: "MULTI_TAP_READOUT_ENABLED",
    16479: "MULTI_TAP_READOUT_SELECTOR",
    16480: "HDR_BRIGHTNESS",
    16481: "DISPLAY_NAME",
    16482: "PIXEL_ALIGNMENT",
    16483: "DECIMATION_MODE",
    16484: "PATH_TO_SAMPLE_IMAGES",
    16485: "ACCEPT_UNEXPECTED_FORMAT",
    16486: "SCAN_COUNT",
    16487: "CUSTOM_FORMAT_XRESOLUTION",
    16488: "CUSTOM_FORMAT_YRESOLUTION",
    16489: "EXPECTED_CONNECTION_INTERFACE",
    16490: "DETECTED_CONNECTION_INTERFACE",
    16491: "AUTO_EXPOSURE_MIN",
    16492: "AUTO_EXPOSURE_MAX",
    16493: "AUTO_GAIN_MAX",
    16494: "SHARPENING_ENABLED",
    16495: "IMAGE_TIMESTAMP",
    16496: "CONVERSION_GAIN_MODE",
    16497: "AUTO_GAIN_ENABLED",
    16498: "AUTO_BRIGHTNESS_MODE",
    16499: "ZOOM",
    16500: "IMAGE_PROCESS_TIME",
    16501: "IMAGE_TRANSFER_TIME",
    16502: "ACQUISITION_FRAMERATE",
    16503: "ACQUISITION_FRAMERATE_ENABLED",
    16504: "HOTPIXEL_CORRECTION_ENABLED",
    16505: "HOTPIXEL_CORRECTION_FACTOR",
}

# PIXEL_TYPE codes for pretty-printing enum option values.
PIXEL_TYPE_NAMES = {
    0x0: "UNKNOWN",
    0x1: "INDEXED",
    0x2: "MONO",
    0x3: "RGB",
    0x4: "BGR",
    0x5: "BAYER_RG",
    0x6: "BAYER_GB",
    0x7: "BAYER_GR",
}


def _pixel_type_str(code: int) -> str:
    base = PIXEL_TYPE_NAMES.get(code & 0xF, f"0x{code & 0xF:X}")
    flags = []
    if code & 0x20000000:
        flags.append("PACKED")
    if code & 0x40000000:
        flags.append(f"EXT:0x{(code >> 16) & 0xFF:X}")
    return base + ("|" + "|".join(flags) if flags else "")


def _try(obj, *names):
    """Call the first existing no-arg method from names; None if all fail."""
    for name in names:
        fn = getattr(obj, name, None)
        if fn is None:
            continue
        with contextlib.suppress(BaseException):
            return fn()
    return None


def _value_of(pv):
    """Extract a plain Python value from a PropertyValue, best-effort."""
    if pv is None:
        return None
    v = _try(pv, "GetValue")
    if v is not None:
        # .NET types repr fine via str(); rects etc fall back to str
        try:
            json.dumps(v)
            return v
        except (TypeError, ValueError):
            return str(v)
    return _try(pv, "ToString")


def probe_property(props_iface, prop_id: int) -> dict | None:
    """Probe one property; returns record dict or None if not implemented."""
    prop = props_iface.FindProperty(prop_id)
    if prop is None:
        return None

    rec: dict = {"id": prop_id, "name": PROP_NAMES.get(prop_id, "?")}

    pv = _try(prop, "GetValue")
    rec["value"] = _value_of(pv)
    idx = _try(pv, "GetIndex")
    if idx is not None:
        rec["index"] = idx

    # PropertyInfo: name candidates across native/.NET wrapper spellings
    info = _try(prop, "GetInfo", "GetPropertyInfo", "getPropertyInfo")
    if info is not None:
        rec["info_class"] = type(info).__name__
        derived = _try(info, "DerivedType", "GetDerivedType", "derivedType", "InfoType", "GetInfoType")
        if derived is not None:
            rec["derived_type"] = str(derived)

        # Enum options
        n = _try(info, "NumOptions", "numOptions", "GetNumOptions")
        if isinstance(n, int) and n > 0:
            options = []
            get_option = None
            for name in ("GetOption", "getOption"):
                if hasattr(info, name):
                    get_option = getattr(info, name)
                    break
            if get_option is not None:
                for i in range(n):
                    with contextlib.suppress(BaseException):
                        options.append(_value_of(get_option(i)))
            rec["options"] = options
            if rec["name"] == "PIXEL_TYPE":
                rec["options_decoded"] = [
                    _pixel_type_str(int(o)) if isinstance(o, (int, float)) else str(o) for o in options
                ]

        # Range min/max/step
        for key, names in [
            ("min", ("MinValue", "minValue", "GetMinValue")),
            ("max", ("MaxValue", "maxValue", "GetMaxValue")),
            ("step", ("StepSize", "stepSize", "GetStepSize")),
        ]:
            v = _value_of(_try(info, *names))
            if v is not None:
                rec[key] = v

    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump K5C UCAPI property space (read-only)")
    parser.add_argument("--json", default=None, help="Also write results to this JSON file")
    parser.add_argument("--reflect", action="store_true", help="Print dir() of first property/info objects")
    args = parser.parse_args()

    conn = LeicaConnection().connect()
    try:
        camera_unit = conn.find_unit(UCAPI_TID.UCAPI_CAMERA)
        if camera_unit is None:
            print("ERROR: camera unit not found")
            return 1
        print(f"Camera unit: {camera_unit.GetName()}")

        # Register UCAPI extensions and init (required before properties
        # are readable; resets camera settings to SDK defaults, as every
        # capture session does). No acquisition follows.
        from LeicaMicrosystems.HardwareModel import Extensions

        with contextlib.suppress(Exception):
            Extensions.ExUCAPI.Register()
        camera_unit.Init()

        props_iface = get_interface(camera_unit, IID.IID_PROPERTIES)
        if props_iface is None:
            print("ERROR: no Properties interface")
            return 1

        n_total = _try(props_iface, "NumProperties")
        print(f"Camera reports {n_total} properties; probing IDs 16384-16505\n")

        if args.reflect:
            prop = props_iface.FindProperty(16399)  # EXPOSURE_TIME
            if prop:
                print("dir(Property):", [m for m in dir(prop) if not m.startswith("_")])
                pv = _try(prop, "GetValue")
                print("dir(PropertyValue):", [m for m in dir(pv) if not m.startswith("_")])
                info = _try(prop, "GetInfo", "GetPropertyInfo")
                if info is not None:
                    print("dir(PropertyInfo):", [m for m in dir(info) if not m.startswith("_")])
                print()

        records = []
        for prop_id in range(16384, 16506):
            rec = probe_property(props_iface, prop_id)
            if rec is None:
                continue
            records.append(rec)
            parts = [f"#{rec['id']} {rec['name']}: value={rec['value']!r}"]
            if "index" in rec:
                parts.append(f"index={rec['index']}")
            if "options" in rec:
                shown = rec.get("options_decoded", rec["options"])
                parts.append(f"options[{len(rec['options'])}]={shown}")
            for k in ("min", "max", "step"):
                if k in rec:
                    parts.append(f"{k}={rec[k]}")
            print("  ".join(parts))

        implemented = {r["id"] for r in records}
        missing = [f"{pid} {PROP_NAMES[pid]}" for pid in PROP_NAMES if pid not in implemented]
        print(f"\nImplemented: {len(records)}  Not implemented: {len(missing)}")
        print("Not implemented:", ", ".join(missing))

        if args.json:
            with open(args.json, "w") as f:
                json.dump({"implemented": records, "not_implemented": missing}, f, indent=2)
            print(f"\nWrote {args.json}")
    finally:
        conn.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
