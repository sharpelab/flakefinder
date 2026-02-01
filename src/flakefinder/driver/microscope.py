import os
from typing import Literal

import clr
import numpy as np

from .helpers import find_unit_by_type
from .enums import (
    IID,
    TID,
    UCAPI_IID,
    UCAPI_PROPERTY,
    UCAPI_UNIT_TYPE,
    EMetricsId,
)

from .interfaces import (
    CancellableImageAcquisitionContext,
    Image,
    ImageAcquisition,
    MetricsConverter,
    Properties,
    Unit,
    AutoCalibration,
)


clr.AddReference(os.path.join(os.path.dirname(__file__), "dlls", "hwmodel2.dll"))
clr.AddReference(os.path.join(os.path.dirname(__file__), "dlls", "hwmodel2exucapi.dll"))
clr.AddReference("System")  # Added for System.Runtime.InteropServices.Marshal

import System  # type: ignore # noqa
from LeicaMicrosystems.HardwareModel import Extensions  # type: ignore # noqa
from LeicaMicrosystems.HardwareModel import HardwareModel  # type: ignore # noqa


class MicroscopeSubunit:
    def __init__(self, microscope_unit: Unit, unit_tid: TID):
        self.microscope_unit = microscope_unit
        self.tid = unit_tid
        self.unit = find_unit_by_type(self.microscope_unit, unit_tid)
        self.name = self.unit.GetName()

    def __repr__(self):
        return f"Name: '{self.name}', Type ID (TID): {self.tid}"


class Axis(MicroscopeSubunit):
    def __init__(self, microscope_unit: Unit, axis_tid: TID):
        super().__init__(microscope_unit, axis_tid)

        self.axis_basic_control_value = (
            self.unit.GetInterfaces()
            .FindInterface(IID.IID_BASIC_CONTROL_VALUE)
            .GetObject()
        )
        self.autocalibration: AutoCalibration = (
            self.unit.GetInterfaces()
            .FindInterface(IID.IID_AUTO_CALIBRATION)
            .GetObject()
        )
        self.metrics_converter = self._get_metrics_converter(EMetricsId.METRICS_MICRONS)

        self.max_native_value = self.axis_basic_control_value.MaxControlValue()
        self.min_native_value = self.axis_basic_control_value.MinControlValue()
        self.max_value = self.native_to_metric(self.max_native_value)
        self.min_value = self.native_to_metric(self.min_native_value)

    def _get_metrics_converter(self, metrics_id: EMetricsId) -> MetricsConverter:
        return (
            self.axis_basic_control_value.GetMetricsConverters().FindMetricsConverter(
                metrics_id
            )
        )

    def native_to_metric(self, native_value: int) -> float:
        return self.metrics_converter.GetMetricsValue(native_value)

    def metric_to_native(self, metric_value: float) -> int:
        return self.metrics_converter.GetControlValue(metric_value)

    @property
    def current_position(self) -> float:
        return self.native_to_metric(self.axis_basic_control_value.GetControlValue())

    @current_position.setter
    def current_position(self, value: float):
        native_value = self.metric_to_native(value)
        self.axis_basic_control_value.SetControlValue(native_value)

    @property
    def current_native_position(self) -> int:
        return self.axis_basic_control_value.GetControlValue()

    @current_native_position.setter
    def current_native_position(self, value: int):
        self.axis_basic_control_value.SetControlValue(value)

    def move_abs(self, metric_value: float) -> int:
        self.current_position = metric_value

    def move_rel(self, metric_value: float) -> int:
        native_value = self.metric_to_native(metric_value)
        self.current_native_position = self.current_native_position + native_value

    def calibrate(self) -> None:
        self.autocalibration.Calibrate()

    @property
    def calibrated(self) -> bool:
        return self.autocalibration.IsCalibrated()


class Stage(MicroscopeSubunit):
    def __init__(self, microscope_unit: Unit, stage_tid: TID = TID.MICROSCOPE_STAGE):
        super().__init__(microscope_unit, stage_tid)

        self.x_axis = Axis(self.microscope_unit, TID.MICROSCOPE_X_UNIT)
        self.y_axis = Axis(self.microscope_unit, TID.MICROSCOPE_Y_UNIT)


class Lamp(MicroscopeSubunit):
    def __init__(self, microscope_unit: Unit, lamp_tid: TID = TID.MICROSCOPE_LAMP):
        super().__init__(microscope_unit, lamp_tid)

        self.intensity_basic_control_value = (
            self.unit.GetInterfaces()
            .FindInterface(IID.IID_BASIC_CONTROL_VALUE)
            .GetObject()
        )

        self.max_intensity = self.intensity_basic_control_value.MaxControlValue()
        self.min_intensity = self.intensity_basic_control_value.MinControlValue()

    @property
    def intensity(self) -> int:
        return self.intensity_basic_control_value.GetControlValue()

    @intensity.setter
    def intensity(self, value: int):
        self.intensity_basic_control_value.SetControlValue(value)

    def off(self):
        self.intensity = 0

    def on(self):
        self.set_default()

    def set_default(self):
        self.intensity = 100


class Camera(MicroscopeSubunit):
    def __init__(
        self,
        microscope_unit: Unit,
        camera_tid: UCAPI_UNIT_TYPE = UCAPI_UNIT_TYPE.UCAPI_CAMERA,
    ):
        super().__init__(microscope_unit, camera_tid)

        self.current_image: np.ndarray = None

        # We need to initialize the camera unit so it can be used
        # The other units dont need this
        self.unit.Init()

        # Set up the image acquisition contexts and handlers
        self.aquisition: ImageAcquisition = (
            self.unit.GetInterfaces()
            .FindInterface(UCAPI_IID.IID_IMAGE_ACQUISITION)
            .GetObject()
        )

        self.aquisition_context: CancellableImageAcquisitionContext = (
            Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory
        )
        self.aquisition_context.ImageAcquiredHandler = (
            Extensions.UCAPI.DelegateOnImageAcquired(self._on_image_acquired)
        )

        # Set up the property handlers
        self.properties: Properties = (
            self.unit.GetInterfaces().FindInterface(IID.IID_PROPERTIES).GetObject()
        )

        self.exposure_time_property_value = self.properties.FindProperty(
            UCAPI_PROPERTY.PROP_EXPOSURE_TIME
        ).GetValue()
        self.auto_brightness_enabled_property_value = self.properties.FindProperty(
            UCAPI_PROPERTY.PROP_AUTO_BRIGHTNESS_ENABLED
        ).GetValue()
        self.gain_property_value = self.properties.FindProperty(
            UCAPI_PROPERTY.PROP_GAIN
        ).GetValue()
        self.gain_blue_property_value = self.properties.FindProperty(
            UCAPI_PROPERTY.PROP_GAIN_BLUE
        ).GetValue()
        self.gain_green_property_value = self.properties.FindProperty(
            UCAPI_PROPERTY.PROP_GAIN_GREEN
        ).GetValue()
        self.gain_red_property_value = self.properties.FindProperty(
            UCAPI_PROPERTY.PROP_GAIN_RED
        ).GetValue()
        self.color_saturation_property_value = self.properties.FindProperty(
            UCAPI_PROPERTY.PROP_COLOUR_SATURATION
        ).GetValue()
        self.gamma_level_property_value = self.properties.FindProperty(
            UCAPI_PROPERTY.PROP_GAMMA_LEVEL
        ).GetValue()
        self.binning_level_property_value = self.properties.FindProperty(
            UCAPI_PROPERTY.PROP_BINNING_LEVEL
        ).GetValue()

        self.set_default()

    @property
    def binning_level(self) -> int:
        return self.binning_level_property_value.GetIndex()

    @binning_level.setter
    def binning_level(self, value: Literal[0, 1, 2]):
        if value not in [0, 1, 2]:
            raise ValueError(
                f"Invalid binning level: {value}. Must be one of [0, 1, 2]."
            )
        self.binning_level_property_value.SetIndex(value)

    @property
    def exposure_time(self) -> float:
        return self.exposure_time_property_value.GetValue()

    @exposure_time.setter
    def exposure_time(self, value: float):
        self.exposure_time_property_value.SetValue(value)

    @property
    def auto_brightness_enabled(self) -> bool:
        return self.auto_brightness_enabled_property_value.GetValue()

    @auto_brightness_enabled.setter
    def auto_brightness_enabled(self, value: bool):
        self.auto_brightness_enabled_property_value.SetValue(value)

    @property
    def gain(self) -> float:
        return self.gain_property_value.GetValue()

    @gain.setter
    def gain(self, value: float):
        self.gain_property_value.SetValue(value)

    @property
    def gain_blue(self) -> float:
        return self.gain_blue_property_value.GetValue()

    @gain_blue.setter
    def gain_blue(self, value: float):
        self.gain_blue_property_value.SetValue(value)

    @property
    def gain_green(self) -> float:
        return self.gain_green_property_value.GetValue()

    @gain_green.setter
    def gain_green(self, value: float):
        self.gain_green_property_value.SetValue(value)

    @property
    def gain_red(self) -> float:
        return self.gain_red_property_value.GetValue()

    @gain_red.setter
    def gain_red(self, value: float):
        self.gain_red_property_value.SetValue(value)

    @property
    def color_saturation(self) -> float:
        return self.color_saturation_property_value.GetValue()

    @color_saturation.setter
    def color_saturation(self, value: float):
        self.color_saturation_property_value.SetValue(value)

    @property
    def gamma_level(self) -> float:
        return self.gamma_level_property_value.GetValue()

    @gamma_level.setter
    def gamma_level(self, value: float):
        self.gamma_level_property_value.SetValue(value)

    @staticmethod
    def _image_to_numpy(image: Image):

        image.LockPixelData()

        try:
            image_format = image.Format()
            width = image_format.Width()
            height = image_format.Height()
            buffer_size = image_format.PixelBufferSize()

            bytes_array = System.Array[System.Byte](buffer_size)

            # Copy the data from the IntPtr to the byte array
            System.Runtime.InteropServices.Marshal.Copy(
                image.PixelData(),
                bytes_array,
                0,
                buffer_size,
            )

            numpy_image = np.frombuffer(bytes_array, dtype=np.uint8).reshape(
                (height, width, -1)
            )

        finally:
            image.UnlockPixelData()

        return numpy_image

    def take_image(self) -> np.ndarray:
        self.aquisition.Acquire(self.aquisition_context, None)
        return self.current_image

    def _on_image_acquired(self, image: Image):
        self.current_image = self._image_to_numpy(image)
        image.Dispose()

    def set_default(self):
        self.auto_brightness_enabled = False
        self.exposure_time = 0.1
        self.gain = 1
        self.gain_blue = 1
        self.gain_green = 1
        self.gain_red = 1
        self.colour_saturation = 100
        self.gamma_level = 1
        self.binning_level = 2

    def __del__(self):
        if self.unit:
            self.unit.Dispose()
            self.aquisition_context.IsCancelled = True
            self.aquisition_context.Dispose()


class Nosepiece(MicroscopeSubunit):
    def __init__(
        self,
        microscope_unit: Unit,
        nosepiece_tid: TID = TID.MICROSCOPE_NOSEPIECE,
    ):
        super().__init__(microscope_unit, nosepiece_tid)

        self.basic_control_value = (
            self.unit.GetInterfaces()
            .FindInterface(IID.IID_BASIC_CONTROL_VALUE)
            .GetObject()
        )

        self.objectives = {"5": 1, "10": 2, "20": 3, "50": 4, "100": 5, "150": 6}
        self.objectives_reversed = {v: k for k, v in self.objectives.items()}

    @property
    def current_objective(self) -> int:
        return self.basic_control_value.GetControlValue()

    @current_objective.setter
    def current_objective(
        self, value: int | Literal["5", "10", "20", "50", "100", "150"]
    ):

        if isinstance(value, str):
            value = self.objectives.get(value)

        if value not in self.objectives.values():
            raise ValueError(
                f"Invalid objective value: {value}. Must be one of {list(self.objectives.values())}."
            )

        self.basic_control_value.SetControlValue(value)


class Aperture(MicroscopeSubunit):
    def __init__(
        self,
        microscope_unit: Unit,
        aperture_tid: TID = TID.MICROSCOPE_IL_APERTURE_DIAPHRAGM,
    ):
        super().__init__(microscope_unit, aperture_tid)

        self.basic_control_value = (
            self.unit.GetInterfaces()
            .FindInterface(IID.IID_BASIC_CONTROL_VALUE)
            .GetObject()
        )

        self.max = self.basic_control_value.MaxControlValue()
        self.min = self.basic_control_value.MinControlValue()

        self.set_default()

    @property
    def current_aperture(self) -> int:
        return self.basic_control_value.GetControlValue()

    @current_aperture.setter
    def current_aperture(self, value: int):
        self.basic_control_value.SetControlValue(value)

    def set_default(self):
        self.current_aperture = self.max


class Shutter(MicroscopeSubunit):
    def __init__(
        self,
        microscope_unit: Unit,
        shutter_tid: TID = TID.MICROSCOPE_IL_SHUTTER,
    ):
        super().__init__(microscope_unit, shutter_tid)

        self.basic_control_value = (
            self.unit.GetInterfaces()
            .FindInterface(IID.IID_BASIC_CONTROL_VALUE)
            .GetObject()
        )

        self.max = self.basic_control_value.MaxControlValue()
        self.min = self.basic_control_value.MinControlValue()

        self.set_default()

    @property
    def current_shutter(self) -> int:
        return self.basic_control_value.GetControlValue()

    @current_shutter.setter
    def current_shutter(self, value: int):
        self.basic_control_value.SetControlValue(value)

    def open(self):
        self.current_shutter = 1

    def close(self):
        self.current_shutter = 0

    def toggle(self):
        if self.current_shutter == 0:
            self.open()
        else:
            self.close()

    def set_default(self):
        self.open()


class Microscope:
    def __init__(
        self,
        config_dir: str = "./",
    ):
        self.hardware_model = HardwareModel.TheHardwareModelInDirectory(config_dir)

        try:
            # We need to register the UCAPI extensions for the camera
            Extensions.ExUCAPI.Register()
        except Exception as e:
            print(f"Error registering UCAPI extensions: {e}")

        self.unit: Unit = self.hardware_model.GetUnit("")
        self.stage = Stage(self.unit)
        self.lamp = Lamp(self.unit)
        self.shutter = Shutter(self.unit)
        self.aperture = Aperture(self.unit)
        self.camera = Camera(self.unit)
        self.nosepiece = Nosepiece(self.unit)
        self.zDrive = Axis(self.unit, TID.MICROSCOPE_ZDRIVE)

    # def __del__(self):
    #     if self.hardware_model:
    #         self.hardware_model.Dispose()
    #         self.hardware_model = None
