"""Probe: dump the DM6M's full AHM unit tree with illumination-state focus.

Read-only audit for the illumination-state investigation (2026-08-13):
two scans with byte-identical recorded settings (lamp/shutter/aperture/
camera) produced very different substrate background colours, so some
illumination-path state is neither pinned nor logged. This probe answers
which illumination units exist on THIS DM6M, which are readable/settable,
and what state each is in right now.

For every unit in the AHM tree it reports: name, type IDs (resolved to
TID names), supported interfaces (resolved to IID names), and — where
BasicControlValue is present — current/min/max control value plus named
positions via DefinedControlValues. Also dumps the microscope root's
ContrastingMethods state (current method + supported method names).

Hardware footprint: opens the SDK connection and READS state only.
NO Init() on any unit, NO SetControlValue, NO camera, NO motion, NO
lamp/shutter changes.

Usage (on the microscope PC; probe outputs go in experiments/):
    uv run python scripts/experiments/probe_illum_units.py --json experiments/illum_units_probe.json
"""

import argparse
import contextlib
import json

from flakefinder.leica.core import LeicaConnection
from flakefinder.leica.enums import IID

# Full microscope TID map (ahwmic.h via docs/sdk_enum_reference.md).
TID_NAMES = {
    0x1000: "MICROSCOPE",
    0x1001: "MICROSCOPE_UNIT",
    0x1002: "MICROSCOPE_MOTORIZED_UNIT",
    0x1003: "MICROSCOPE_CODED_UNIT",
    0x1004: "MICROSCOPE_NOSEPIECE",
    0x1005: "MICROSCOPE_ZDRIVE",
    0x1006: "MICROSCOPE_TL_AXIS",
    0x1007: "MICROSCOPE_IL_AXIS",
    0x1008: "MICROSCOPE_IL_TURRET",
    0x1009: "MICROSCOPE_APERTURE_DIAPHRAGM",
    0x100A: "MICROSCOPE_FIELD_DIAPHRAGM",
    0x100B: "MICROSCOPE_CONDENSER",
    0x100C: "MICROSCOPE_CONDENSER_TURRET",
    0x100D: "MICROSCOPE_SWITCHABLE_CONDENSER_TOP",
    0x100E: "MICROSCOPE_FUNCTION_KEYS",
    0x100F: "MICROSCOPE_POLARIZER",
    0x1010: "MICROSCOPE_STAGE",
    0x1011: "MICROSCOPE_LENS",
    0x1012: "MICROSCOPE_TL_ADAPTING_LENS",
    0x1013: "MICROSCOPE_SHUTTER",
    0x1014: "MICROSCOPE_DIC_TURRET",
    0x1015: "MICROSCOPE_SCREEN",
    0x1016: "MICROSCOPE_TOUCHSCREEN",
    0x1017: "MICROSCOPE_LAMP",
    0x1018: "MICROSCOPE_TL_IL_LAMP_SWITCH",
    0x1019: "MICROSCOPE_MAGNIFICATION_CHANGER",
    0x101A: "MICROSCOPE_X_UNIT",
    0x101B: "MICROSCOPE_Y_UNIT",
    0x101C: "MICROSCOPE_Z_UNIT",
    0x101D: "MICROSCOPE_COMPARISON_BRIDGE",
    0x101E: "MICROSCOPE_LEFT_STAND",
    0x101F: "MICROSCOPE_RIGHT_STAND",
    0x1020: "MICROSCOPE_OBSERVATION_DIAPHRAGM",
    0x1021: "MICROSCOPE_COLOUR_COMPENSATION",
    0x1022: "MICROSCOPE_TURRET",
    0x1023: "MICROSCOPE_INCIDENT_LIGHT_UNIT",
    0x1024: "MICROSCOPE_TRANSMITTED_LIGHT_UNIT",
    0x1025: "MICROSCOPE_TL_SHUTTER",
    0x1026: "MICROSCOPE_IL_SHUTTER",
    0x1028: "MICROSCOPE_DIAPHRAGM",
    0x1029: "MICROSCOPE_IRIS_DIAPHRAGM",
    0x102A: "MICROSCOPE_MOTORZOOM",
    0x102B: "MICROSCOPE_FLUORESCENCE_UNIT",
    0x102C: "MICROSCOPE_TL_APERTURE_DIAPHRAGM",
    0x102D: "MICROSCOPE_TL_FIELD_DIAPHRAGM",
    0x102E: "MICROSCOPE_IL_APERTURE_DIAPHRAGM",
    0x102F: "MICROSCOPE_IL_FIELD_DIAPHRAGM",
    0x1030: "MICROSCOPE_IL_ATTENUATOR",
    0x1031: "MICROSCOPE_PORTS",
    0x1032: "MICROSCOPE_TL_POLARIZER",
    0x1033: "MICROSCOPE_MANUAL_UNIT",
    0x1034: "MICROSCOPE_MANUAL_DIC_SLIDER",
    0x1035: "MICROSCOPE_SLIDER",
    0x1036: "MICROSCOPE_EXTERNAL_UNIT",
    0x1037: "MICROSCOPE_CONTROLLER_BOARD",
    0x1038: "MICROSCOPE_STAGE_LR_SYNC",
    0x1039: "MICROSCOPE_ZDRIVE_LR_SYNC",
    0x103A: "MICROSCOPE_COAXIAL_LIGHT_UNIT",
    0x103B: "MICROSCOPE_OBLIQUE_LIGHT_UNIT",
    0x103C: "MICROSCOPE_MEMORY_FUNCTION_UNIT",
    0x103D: "MICROSCOPE_VIRTUAL_UNIT",
    0x103E: "MICROSCOPE_ZDRIVE_FINE",
    0x1040: "MICROSCOPE_SIDEPORTS",
    0x1041: "MICROSCOPE_BOTTOMPORT",
    0x1042: "MICROSCOPE_SCREEN_CONTRAST",
    0x1043: "MICROSCOPE_SCREEN_ILLUMINATION",
    0x1044: "MICROSCOPE_IL_EXCITATION_MANAGER",
    0x1045: "MICROSCOPE_IL_FAST_FILTERS",
    0x1046: "MICROSCOPE_TL_FILTER",
    0x1047: "MICROSCOPE_TL_FILTER1",
    0x1048: "MICROSCOPE_TL_FILTER2",
    0x1049: "MICROSCOPE_TL_MIRROR",
    0x104A: "MICROSCOPE_IL_MIRROR",
    0x104B: "MICROSCOPE_MIRROR",
    0x104C: "MICROSCOPE_TIRF_UNIT",
    0x104D: "MICROSCOPE_TIRF_POSITIONER",
    0x104E: "MICROSCOPE_TIRF_COLLIMATOR",
    0x104F: "MICROSCOPE_TIRF_LASER",
    0x1050: "MICROSCOPE_LASER",
    0x1051: "MICROSCOPE_TIRF_SAFETY_ATTENUATOR",
    0x1052: "MICROSCOPE_TIRF_SHUTTER",
    0x1053: "MICROSCOPE_TL_ATTENUATOR",
    0x1054: "MICROSCOPE_TL_ISOCOL",
    0x1055: "MICROSCOPE_LIGHT_UNIT",
    0x1056: "MICROSCOPE_FUNCTION_WHEELS",
    0x1057: "MICROSCOPE_HAND_CONTROL",
    0x1058: "MICROSCOPE_STAGE_WELL_HANDLING",
    0x1059: "MICROSCOPE_DUOPORT",
    0x105A: "MICROSCOPE_VISUAL_ZOOM",
    0x105B: "MICROSCOPE_VIDEO_ZOOM",
    0x105C: "MICROSCOPE_IL_FAST_FILTER_WHEELS",
    0x105D: "MICROSCOPE_IL_FAST_FILTER_WHEEL",
    0x105E: "MICROSCOPE_IL_FAST_FILTER_WHEEL_ATTENUATOR",
    0x105F: "MICROSCOPE_IL_FAST_FILTER_WHEEL_EXCITER",
    0x1060: "MICROSCOPE_IL_FAST_FILTER_WHEEL_EMITTER",
    0x1061: "MICROSCOPE_FOOT_CONTROL",
    0x1062: "MICROSCOPE_CLS_SHUTTER",
    0x1063: "MICROSCOPE_CONOSCOPY_UNIT",
    0x1064: "MICROSCOPE_PORT_MAGNIFICATION_CHANGER",
    0x1065: "MICROSCOPE_TL_CCIC_FILTER",
    0x1066: "MICROSCOPE_UNIVERSAL_MANUAL_CONTROL",
    0x1067: "MICROSCOPE_JOYSTICK",
    0x1068: "MICROSCOPE_SMARTMOVE",
    0x1069: "MICROSCOPE_SEQUENCER",
    0x106A: "MICROSCOPE_SEQUENCING",
    0x106B: "MICROSCOPE_SDCONFOCAL_UNIT",
    0x106C: "MICROSCOPE_SDCONFOCAL_EXCITATION_WHEEL",
    0x106D: "MICROSCOPE_SDCONFOCAL_EMISSION_WHEEL",
    0x106E: "MICROSCOPE_SDCONFOCAL_DICHROIC_WHEEL",
    0x106F: "MICROSCOPE_SDCONFOCAL_DISK_SLIDER",
    0x1070: "MICROSCOPE_SDCONFOCAL_SHUTTER",
    0x1071: "MICROSCOPE_SDCONFOCAL_PRISM_SLIDER",
    0x1072: "MICROSCOPE_SDCONFOCAL_FRAP_IRIS",
    0x1073: "MICROSCOPE_SDCONFOCAL_VAR_INTENSITY_IRIS",
    0x1074: "MICROSCOPE_SDCONFOCAL_ILTURRET",
    0x1075: "MICROSCOPE_COLD_LIGHT_SOURCE",
    0x1076: "MICROSCOPE_CLS_LAMP",
    0x1077: "MICROSCOPE_IL_LAMP",
    0x1078: "MICROSCOPE_TL_LAMP",
    0x1079: "MICROSCOPE_OL_LAMP",
    0x107A: "MICROSCOPE_CL_LAMP",
    0x107B: "MICROSCOPE_CL_SHUTTER",
    0x107C: "MICROSCOPE_OL_SHUTTER",
    0x107D: "MICROSCOPE_ARCLIGHT_UNIT",
    0x107E: "MICROSCOPE_ARCLIGHT_INTENSITY",
    0x107F: "MICROSCOPE_ARCLIGHT_SHUTTER",
    0x1080: "MICROSCOPE_ARCLIGHT_SCENE",
    0x1081: "MICROSCOPE_RINGLIGHT_UNIT",
    0x1082: "MICROSCOPE_RINGLIGHT_INTENSITY",
    0x1083: "MICROSCOPE_RINGLIGHT_SHUTTER",
    0x1084: "MICROSCOPE_RINGLIGHT_SCENE",
    0x1085: "MICROSCOPE_SDCONFOCAL_SPINNING_DISK",
    0x1086: "MICROSCOPE_SEQUENCER_TIMESTAMPS",
    0x1087: "MICROSCOPE_CAMERA_INTERFACE",
    0x1088: "MICROSCOPE_FILES",
    0x1089: "MICROSCOPE_IL_LEDS",
    0x108A: "MICROSCOPE_IL_LED",
    0x108B: "MICROSCOPE_LEDCXI_UNIT",
    0x108C: "MICROSCOPE_LEDCXI_INTENSITY",
    0x108D: "MICROSCOPE_LEDCXI_SHUTTER",
    0x108E: "MICROSCOPE_LEDCXI_SCENE",
    0x108F: "MICROSCOPE_LEDNVI_UNIT",
    0x1090: "MICROSCOPE_LEDNVI_INTENSITY",
    0x1091: "MICROSCOPE_LEDNVI_SHUTTER",
    0x1092: "MICROSCOPE_LEDNVI_SCENE",
    0x1093: "MICROSCOPE_SMARTTOUCH",
    0x1094: "MICROSCOPE_ZOOM_DISPLAY",
    0x1095: "MICROSCOPE_DATA_CONVERSION_INTERFACE",
    0x1096: "MICROSCOPE_IL_LEDS_SLIDER",
    0x1099: "MICROSCOPE_LED_UNIT",
    0x109A: "MICROSCOPE_LED_INTENSITY",
    0x109B: "MICROSCOPE_LED_SHUTTER",
    0x109C: "MICROSCOPE_LED_SCENE",
    0x109F: "MICROSCOPE_ROTARY_HEAD",
    0x10A0: "MICROSCOPE_LEDHDI_UNIT",
    0x10A1: "MICROSCOPE_LEDHDI_INTENSITY",
    0x10A2: "MICROSCOPE_LEDHDI_SHUTTER",
    0x10A3: "MICROSCOPE_LEDHDI_SCENE",
    0x10A4: "MICROSCOPE_DVMLED1_UNIT",
    0x10A5: "MICROSCOPE_DVMLED1_INTENSITY",
    0x10A6: "MICROSCOPE_DVMLED1_SHUTTER",
    0x10A7: "MICROSCOPE_DVMLED2_UNIT",
    0x10A8: "MICROSCOPE_DVMLED2_INTENSITY",
    0x10A9: "MICROSCOPE_DVMLED2_SHUTTER",
    0x10AA: "MICROSCOPE_SEQUENCER_REALTIME_TIMESTAMPS",
    0x10AB: "MICROSCOPE_IL_STRUCTURED_ILLUMINATION",
    0x10AC: "MICROSCOPE_IL_STRUCTURED_ILLUMINATION_APERTURE_DIAPHRAGM",
    0x10AE: "MICROSCOPE_WATERPUMP",
    0x10AF: "MICROSCOPE_IL_UV_SHUTTER",
    0x10B6: "MICROSCOPE_TL_CONTRASTING_TUNING",
    0x10B7: "MICROSCOPE_TL_TILT",
    0x10B8: "MICROSCOPE_TL_INTENSITY",
    0x10B9: "MICROSCOPE_NOSEPIECE_MOTCORR",
    0x10BA: "MICROSCOPE_CONTRASTING_METHODS_AUTOMATED",
    0x10BB: "MICROSCOPE_VIDEO_DEPTH_OF_FIELD_UNIT",
    0x10BC: "MICROSCOPE_VIDEO_RESOLUTION_UNIT",
    0x10BD: "MICROSCOPE_VIDEO_FIELD_OF_VIEW_WIDTH_UNIT",
    0x10BE: "MICROSCOPE_VIDEO_FIELD_OF_VIEW_HEIGHT_UNIT",
    0x10BF: "MICROSCOPE_VISUAL_DEPTH_OF_FIELD_UNIT",
    0x10C0: "MICROSCOPE_VISUAL_RESOLUTION_UNIT",
    0x10C1: "MICROSCOPE_VISUAL_FIELD_OF_VIEW_UNIT",
    0x10C2: "MICROSCOPE_LEDSLI_UNIT",
    0x10C3: "MICROSCOPE_LEDSLI_INTENSITY",
    0x10C4: "MICROSCOPE_LEDSLI_SHUTTER",
    0x10C5: "MICROSCOPE_LEDSLI_SCENE",
    0x10C6: "MICROSCOPE_ACTIVE_UNIT",
    0x10C7: "MICROSCOPE_NUMERICAL_APERTURE_UNIT",
    0x10C9: "MICROSCOPE_HAND_CONTROL_Z",
    0x10CA: "MICROSCOPE_BACKLIGHT_INTENSITY",
    0x10CB: "MICROSCOPE_BACKLIGHT_SHUTTER",
    0x10CC: "MICROSCOPE_POSITIONS_MEMORY",
    0x10CD: "MICROSCOPE_DVM",
    0x10CE: "MICROSCOPE_DMS",
    0x10CF: "MICROSCOPE_GENERIC",
    0x10D0: "MICROSCOPE_CLIMATE_CONTROL",
    0x10D1: "MICROSCOPE_CLIMATE_UNIT",
    0x10D2: "MICROSCOPE_ZDRIVE_CLOSED_LOOP",
    0x10D3: "MICROSCOPE_LASER_DUAL_SHUTTER",
    0x10D4: "MICROSCOPE_STAGE_ROTATION_UNIT",
    0x10D5: "MICROSCOPE_COLUMN_TILT_UNIT",
    0x10D6: "MICROSCOPE_SIMULTANEOUS_IMAGING_SYSTEM",
    0x10D7: "MICROSCOPE_FUNCTION_KEYS_LEDS",
    0x10D8: "MICROSCOPE_LEDBLI_UNIT",
    0x10D9: "MICROSCOPE_LEDBLI_INTENSITY",
    0x10DA: "MICROSCOPE_LEDBLI_SHUTTER",
    0x10DB: "MICROSCOPE_LEDDI_UNIT",
    0x10DC: "MICROSCOPE_LEDDI_INTENSITY",
    0x10DD: "MICROSCOPE_LEDDI_SHUTTER",
    0x10DF: "MICROSCOPE_SPIM_OBJECTIVE_HOLDER",
    0x10E0: "MICROSCOPE_SPIM_FILTER_TURRET",
    0x10E1: "MICROSCOPE_SPIM_OBJECTIVE_TURRET",
    0x10E2: "MICROSCOPE_T_HOUSE",
    0x10E3: "MICROSCOPE_TESTABLE_KEYS",
    0x10E4: "MICROSCOPE_ACQUIMATOR",
    0x10E5: "MICROSCOPE_DVM_AUTOFOCUS",
    0x10E6: "MICROSCOPE_FRAP",
    0x10E7: "MICROSCOPE_FRAP_DIAPHRAGM",
    0x10E8: "MICROSCOPE_FRAP_LASER",
    0x10E9: "MICROSCOPE_IL_UV_LAMP",
    0x10EA: "MICROSCOPE_RFID_UNIT",
    0x10EB: "MICROSCOPE_SEQUENCER_ADVANCED",
    0x10EC: "MICROSCOPE_SEQUENCER_ADVANCED_UNIT",
    0x10ED: "MICROSCOPE_SEQUENCER_ADVANCED_TTL",
    0x10EE: "MICROSCOPE_SEQUENCER_ADVANCED_ANALOG",
    0x10EF: "MICROSCOPE_SEQUENCER_ADVANCED_LASERBOX",
    0x10F0: "MICROSCOPE_FIBERSWITCH",
    0x10F1: "MICROSCOPE_SEQUENCER_ADVANCED_LASER",
    0x10F2: "MICROSCOPE_SEQUENCER_ADVANCED_INPUT",
    0x10F3: "MICROSCOPE_SEQUENCER_ADVANCED_OUTPUT",
    0x10F4: "MICROSCOPE_SEQUENCER_ADVANCED_LASERBOX_GENERIC",
    0x10F5: "MICROSCOPE_SEQUENCER_ADVANCED_TIRF_LASER",
    0x10F6: "MICROSCOPE_SCANHEAD",
    0x10F7: "MICROSCOPE_SCANHEAD_DIAPHRAGM",
    0x10F8: "MICROSCOPE_SCANHEAD_ATTENUATOR",
    0x10F9: "MICROSCOPE_SCANHEAD_AFOCAL_LENS",
    0x10FA: "MICROSCOPE_SCANHEAD_SHIFT_PRISM",
    0x10FB: "MICROSCOPE_INFINITY_SCANNER",
    0x10FC: "MICROSCOPE_LASER_SPECTROSCOPY",
    0x10FD: "MICROSCOPE_SEQUENCER_ADVANCED_LASER_DUAL_SHUTTER",
    0x10FE: "MICROSCOPE_BARCODE_SCANNER",
    0x10FF: "MICROSCOPE_IL_LAMP_SWITCH",
    0x1100: "MICROSCOPE_SEQUENCER_ADVANCED_PULSED_LASER",
    0x1101: "MICROSCOPES",
    0x1102: "MICROSCOPE_LIGHT_GUARD",
    0x1103: "MICROSCOPE_BARCODE_SCANNERS",
    0x1104: "MICROSCOPE_SEQUENCER_ADVANCED_PULSED_LASER_GUARD",
    0x1105: "MICROSCOPE_SEQUENCER_ADVANCED_CAMERA_TRIGGER",
    0x1106: "MICROSCOPE_POWER_SWITCH_CAMERA",
    0x1107: "MICROSCOPE_POWER_SWITCH_BARCODE_SCANNER",
    0x1108: "MICROSCOPE_SEQUENCER_ADVANCED_INTERNAL_TRIGGER",
    0x1109: "MICROSCOPE_IL_TURRET_FLAP",
    0x110A: "MICROSCOPE_IMC",
    0x110B: "MICROSCOPE_INTEGRATED_WATERPUMP",
    0x110C: "MICROSCOPE_ACCESS_DOOR",
    0x110D: "MICROSCOPE_STATUS_INDICATOR",
    0x110E: "MICROSCOPE_INTEGRATED_USBHUB",
    0x1110: "MICROSCOPE_INTERIOR_LIGHT",
    0x1111: "MICROSCOPE_IL_LEDS_MODULE_SHUTTER",
    0x1112: "MICROSCOPE_IMC_X",
    0x1113: "MICROSCOPE_IMC_Z",
    0x1114: "MICROSCOPE_INSPECTION_HATCH",
    0x1115: "MICROSCOPE_INTERLOCK_CIRCUIT",
    0x1116: "MICROSCOPE_INTEGRATED_WATERPUMP_SENSOR",
    0x1117: "MICROSCOPE_TEMPERATURE_CONTROL",
    0x1118: "MICROSCOPE_FAN",
    0x1119: "MICROSCOPE_TEMPERATURE_SENSOR",
    0x111C: "MICROSCOPE_IL_LEDS_MODULE_INTENSITY_SENSOR",
}

# IID map (ahwbasic.h + ahwmic.h via docs/sdk_enum_reference.md).
IID_NAMES = {
    0x0010: "IID_DRIVER_INFO",
    0x0011: "IID_SESSION",
    0x0012: "IID_ALL_SESSIONS",
    0x0013: "IID_RESERVED",
    0x0014: "IID_INIT_STATE",
    0x0015: "IID_DYNAMIC_UNITS",
    0x0016: "IID_ORIGINAL_DRIVER_INFO",
    0x0100: "IID_BASIC_CONTROL_VALUE",
    0x0101: "IID_BASIC_CALIBRATION",
    0x0102: "IID_AUTO_CALIBRATION",
    0x0103: "IID_BASIC_CONFIGURATION",
    0x0104: "IID_PROPERTIES",
    0x0105: "IID_SNAPSHOTS",
    0x0106: "IID_EVENT_SOURCE",
    0x0107: "IID_DELTA_CONTROL_VALUE",
    0x0108: "IID_BASIC_CONTROL_VALUE_VELOCITY",
    0x0109: "IID_LIMIT_CONTROL_VALUE",
    0x010A: "IID_BASIC_CONTROL_STATE",
    0x010B: "IID_HALT_CONTROL_VALUE",
    0x010C: "IID_DIRECTED_CONTROL_VALUE_ASYNC",
    0x010D: "IID_BASIC_CONTROL_VALUE_ASYNC",
    0x010E: "IID_FACTORY_SETTINGS",
    0x010F: "IID_ADVANCED_EVENT_SOURCE",
    0x0110: "IID_DEFINED_CONTROL_VALUES",
    0x0111: "IID_BASIC_CONTROL_VALUE_SPECIAL",
    0x0112: "IID_PROPERTIES_UI",
    0x0113: "IID_ALTERNATING_CONTROL_UNITS",
    0x0114: "IID_DIRECTED_CONTROL_VALUE_ASYNC_VELOCITY",
    0x0115: "IID_LIMIT_SWITCHES",
    0x0116: "IID_LOGGING",
    0x0117: "IID_BASIC_CONTROL_VALUE_TIMING",
    0x0118: "IID_BASIC_CONTROL_VALUE_SETS",
    0x0119: "IID_BASIC_CONTROL_VALUE_HYSTERESIS_CORRECTED",
    0x011A: "IID_PROPERTY_ID_NAMES",
    0x011B: "IID_DELTA_CONTROL_VALUE_ASYNC",
    0x011C: "IID_BASIC_CONTROL_VALUE_ACCELERATION",
    0x011D: "IID_DELTA_CONTROL_VALUE_SPECIAL",
    0x011E: "IID_RESERVED_DIAGNOSTICS",
    0x0120: "IID_ORIGINAL_PROPERTIES_DRIVER_INFO",
    0x0121: "IID_RENDER2D",
    0x0122: "IID_RENDER2D_CALIBRATION",
    0x0123: "IID_RENDER2D_XML_PROCESSING",
    0x0125: "IID_BASIC_ACTION",
    0x1000: "IID_MICROSCOPE_CONTRASTING_METHODS",
    0x1002: "IID_MICROSCOPE_RESERVED0",
    0x1003: "IID_MICROSCOPE_MEMORY",
    0x1004: "IID_MICROSCOPE_CONTRASTING_METHODS_LOOKAHEAD",
    0x1005: "IID_MICROSCOPE_RESERVED1",
    0x1006: "IID_MICROSCOPE_SHEARING",
    0x1007: "IID_MICROSCOPE_IL_SHEARING",
    0x1008: "IID_MICROSCOPE_TIRF_CALIBRATION",
    0x1009: "IID_MICROSCOPE_TIRF_AZIMUTH",
    0x100A: "IID_MICROSCOPE_TIRF_APERTURE",
    0x100C: "IID_MICROSCOPE_IL_FAST_FILTER_WHEEL_CALIBRATION",
    0x100D: "IID_MICROSCOPE_REFLECTION",
    0x1010: "IID_MICROSCOPE_CALIBRATION_TABLE",
    0x1011: "IID_MICROSCOPE_SCENE_SEQUENCE",
    0x1012: "IID_MICROSCOPE_SEQUENCER_TIMESTAMPS",
    0x1013: "IID_MICROSCOPE_SEQUENCER_CONTROL",
    0x1014: "IID_MICROSCOPE_TIRF_SERVICE",
    0x1015: "IID_MICROSCOPE_FILES_OBJECTIVE",
    0x1016: "IID_MICROSCOPE_ZOOM_CALIBRATION",
    0x1017: "IID_MICROSCOPE_EXACT_ZFOCUS",
    0x101A: "IID_MICROSCOPE_SEQUENCER_TRIGGERING_SETUP",
    0x101C: "IID_MICROSCOPE_DVM_AUTOFOCUS",
    0x101D: "IID_MICROSCOPE_IL_TURRET_RFID_SETUP",
    0x101E: "IID_MICROSCOPE_CONTRASTING_METHODS_UPDATE",
    0x101F: "IID_MICROSCOPE_SEQADV_CONTROL",
    0x1020: "IID_MICROSCOPE_SEQADV_CONTROL_ERROR",
    0x1023: "IID_MICROSCOPE_BARCODE_SCANNER",
    0x1024: "IID_MICROSCOPE_HYDRA_SEQADV",
    0x1025: "IID_MICROSCOPE_WATERPUMP_AUTOIMMERSION",
    0x1026: "IID_MICROSCOPE_IL_LEDS_INTENSITY_REF_MEASUREMENT",
    0x1027: "IID_MICROSCOPE_IMC_XZ_PUPIL_CONTROL",
    0x1028: "IID_MICROSCOPE_DYNCOLLSION_INFOS",
    0x1022: "IID_MICROSCOPE_LASER_SPECTROSCOPY",
    0x3100: "IID_AUTOFOCUS",
    0x3101: "IID_DCAUTOFOCUS_INIT",
    0x3102: "IID_CMS_AUTOFOCUS",
}

# TIDs whose state shapes illumination colour/intensity — flagged in the
# summary table. Everything else is still dumped in the JSON.
ILLUM_TIDS = {
    0x1008,  # IL_TURRET (reflector / filter cube)
    0x1009,  # APERTURE_DIAPHRAGM (generic)
    0x100A,  # FIELD_DIAPHRAGM (generic)
    0x100B,  # CONDENSER
    0x100C,  # CONDENSER_TURRET
    0x100D,  # SWITCHABLE_CONDENSER_TOP
    0x1014,  # DIC_TURRET
    0x1017,  # LAMP
    0x1018,  # TL_IL_LAMP_SWITCH
    0x1020,  # OBSERVATION_DIAPHRAGM
    0x1022,  # TURRET (generic)
    0x1025,  # TL_SHUTTER
    0x1026,  # IL_SHUTTER
    0x1028,  # DIAPHRAGM (generic)
    0x1029,  # IRIS_DIAPHRAGM
    0x102C,  # TL_APERTURE_DIAPHRAGM
    0x102D,  # TL_FIELD_DIAPHRAGM
    0x102E,  # IL_APERTURE_DIAPHRAGM
    0x102F,  # IL_FIELD_DIAPHRAGM
    0x1030,  # IL_ATTENUATOR
    0x1031,  # PORTS
    0x1053,  # TL_ATTENUATOR
    0x1055,  # LIGHT_UNIT
    0x1077,  # IL_LAMP
    0x1078,  # TL_LAMP
    0x10FF,  # IL_LAMP_SWITCH
    0x1109,  # IL_TURRET_FLAP
}


def _try(obj, *names):
    """Call the first existing no-arg method from names; None if all fail."""
    for name in names:
        fn = getattr(obj, name, None)
        if fn is None:
            continue
        with contextlib.suppress(BaseException):
            return fn()
    return None


def _try_call(obj, names, *args):
    """Call the first existing method from names with args; None if all fail."""
    for name in names:
        fn = getattr(obj, name, None)
        if fn is None:
            continue
        with contextlib.suppress(BaseException):
            return fn(*args)
    return None


def _type_ids(unit) -> list[int]:
    """All type IDs a unit exposes (most-derived first is not guaranteed)."""
    ut = unit.GetUnitType()
    ids = []
    n = _try(ut, "NumTypeIds")
    if n:
        for i in range(n):
            tid = _try_call(ut, ["GetTypeId"], i)
            if tid is not None:
                ids.append(int(tid))
    exposed = _try(ut, "ExposedTypeId")
    if exposed is not None and int(exposed) not in ids:
        ids.insert(0, int(exposed))
    return ids


def _iface_ids(unit) -> list[int]:
    ifaces = unit.GetInterfaces()
    if ifaces is None:
        return []
    ids = []
    n = _try(ifaces, "NumInterfaces") or 0
    for i in range(n):
        iface = _try_call(ifaces, ["GetInterface"], i)
        if iface is None:
            continue
        iid = _try(iface, "GetInterfaceId")
        if iid is not None:
            ids.append(int(iid))
    return ids


def _get_iface(unit, iid: int):
    ifaces = unit.GetInterfaces()
    if ifaces is None:
        return None
    iface = _try_call(ifaces, ["FindInterface"], int(iid))
    if iface is None:
        return None
    return _try(iface, "GetObject")


def _bcv_state(unit) -> dict | None:
    bcv = _get_iface(unit, IID.IID_BASIC_CONTROL_VALUE)
    if bcv is None:
        return None
    state = {
        "value": _try(bcv, "GetControlValue"),
        "min": _try(bcv, "MinControlValue"),
        "max": _try(bcv, "MaxControlValue"),
    }
    return state


def _defined_values(unit) -> list[dict] | None:
    dcv = _get_iface(unit, IID.IID_DEFINED_CONTROL_VALUES)
    if dcv is None:
        return None
    names = _try(dcv, "DefinedNames", "definedNames")
    n = _try(dcv, "NumDefinedControlValues", "numDefinedControlValues")
    if n is None:
        return None
    out = []
    for i in range(n):
        dv = _try_call(dcv, ["GetDefinedControlValue", "getDefinedControlValue"], i)
        if dv is None:
            continue
        rec = {
            "id": _try(dv, "Id", "GetId", "id"),
            "value": _try(dv, "DefinedValue", "definedValue"),
        }
        if names is not None and rec["id"] is not None:
            rec["name"] = _try_call(names, ["FindName", "findName"], rec["id"])
        out.append(rec)
    return out


def _properties(unit) -> list[dict] | None:
    props = _get_iface(unit, IID.IID_PROPERTIES)
    if props is None:
        return None
    n = _try(props, "NumProperties") or 0
    out = []
    for i in range(n):
        prop = _try_call(props, ["GetProperty"], i)
        if prop is None:
            continue
        pv = _try(prop, "GetValue")
        val = _try(pv, "GetValue") if pv is not None else None
        try:
            json.dumps(val)
        except (TypeError, ValueError):
            val = str(val)
        rec = {"id": _try(prop, "GetId"), "value": val}
        idx = _try(pv, "GetIndex") if pv is not None else None
        if idx is not None:
            rec["index"] = idx
        out.append(rec)
    return out


def _id_list(obj) -> list[int]:
    if obj is None:
        return []
    n = _try(obj, "NumIds", "numIds") or 0
    return [v for i in range(n) if (v := _try_call(obj, ["GetId", "getId"], i)) is not None]


def _contrasting_methods(unit) -> dict | None:
    cm = _get_iface(unit, IID.IID_MICROSCOPE_CONTRASTING_METHODS)
    if cm is None:
        return None
    names = _try(cm, "MethodNames", "methodNames")

    def _named(ids):
        out = []
        for mid in ids:
            name = _try_call(names, ["FindName", "findName"], mid) if names else None
            out.append({"id": mid, "name": name})
        return out

    current = _try(cm, "GetContrastingMethod", "getContrastingMethod")
    current_name = None
    if names is not None and current is not None:
        current_name = _try_call(names, ["FindName", "findName"], current)
    rec = {
        "current": current,
        "current_name": current_name,
        "supported": _named(_id_list(_try(cm, "SupportedMethods", "supportedMethods"))),
        "all": _named(_id_list(_try(cm, "AllMethods", "allMethods"))),
    }
    return rec


def walk(unit, path: str, records: list[dict]) -> None:
    name = _try(unit, "GetName") or "?"
    tids = _type_ids(unit)
    iids = _iface_ids(unit)
    rec = {
        "path": path,
        "name": name,
        "type_ids": [f"0x{t:04X}" for t in tids],
        "type_names": [TID_NAMES.get(t, f"0x{t:04X}") for t in tids],
        "interfaces": [IID_NAMES.get(i, f"0x{i:04X}") for i in iids],
        "illumination": any(t in ILLUM_TIDS for t in tids),
    }
    state = _bcv_state(unit)
    if state is not None:
        rec["control"] = state
    defined = _defined_values(unit)
    if defined:
        rec["defined_values"] = defined
    props = _properties(unit)
    if props:
        rec["properties"] = props
    cm = _contrasting_methods(unit)
    if cm is not None:
        rec["contrasting_methods"] = cm
    records.append(rec)

    children = unit.GetUnits()
    if children is None:
        return
    n = _try(children, "NumUnits") or 0
    for i in range(n):
        child = _try_call(children, ["GetUnit"], i)
        if child is None:
            continue
        walk(child, f"{path}/{i}", records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only dump of the DM6M AHM unit tree (illumination focus)")
    parser.add_argument("--json", type=str, default=None, help="Write full dump to this JSON path")
    args = parser.parse_args()

    conn = LeicaConnection()
    conn.connect()
    try:
        records: list[dict] = []
        walk(conn.root, "root", records)
    finally:
        conn.disconnect()

    # Summary: illumination-flagged units + any unit with a control value
    print(f"{len(records)} units in tree\n")
    print(f"{'unit':<32} {'type':<36} {'value':>18}  defined/notes")
    print("-" * 110)
    for rec in records:
        interesting = rec["illumination"] or "control" in rec or "contrasting_methods" in rec
        if not interesting:
            continue
        tname = rec["type_names"][0] if rec["type_names"] else "?"
        ctrl = rec.get("control")
        vstr = f"{ctrl['value']} [{ctrl['min']}-{ctrl['max']}]" if ctrl else "-"
        notes = []
        if rec["illumination"]:
            notes.append("ILLUM")
        for dv in rec.get("defined_values", []) or []:
            notes.append(f"{dv.get('name') or dv.get('id')}={dv.get('value')}")
        cm = rec.get("contrasting_methods")
        if cm:
            sup = ", ".join(f"{m['name'] or m['id']}" for m in cm["supported"])
            notes.append(f"method={cm['current_name'] or cm['current']} of [{sup}]")
        print(f"{rec['name'][:32]:<32} {tname[:36]:<36} {vstr:>18}  {'; '.join(notes)}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"units": records}, f, indent=2)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
