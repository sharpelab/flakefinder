from abc import ABC, abstractmethod

from .enums import (
    IID,
    IMAGE_ORIENTATION,
    PIXEL_FORMAT,
    EMetricsId,
)


class Unit(ABC):
    @abstractmethod
    def GetName(self) -> str:
        pass

    @abstractmethod
    def Dispose(self) -> None:
        pass

    @abstractmethod
    def GetUnitType(self) -> "UnitType":
        pass

    @abstractmethod
    def GetUnits(self) -> "Units":
        pass

    @abstractmethod
    def GetParent(self) -> "Unit":
        pass

    @abstractmethod
    def GetInterfaces(self) -> "Interfaces":
        pass

    @abstractmethod
    def Init(self) -> None:
        pass


class UnitType(ABC):
    @abstractmethod
    def ExposedTypeId(self) -> int:
        pass

    @abstractmethod
    def IsA(self, typeId: int) -> bool:
        pass

    @abstractmethod
    def NumTypeIds(self) -> int:
        pass

    @abstractmethod
    def GetTypeId(self, index: int) -> int:
        pass


class Units(ABC):
    @abstractmethod
    def NumUnits(self) -> int:
        pass

    @abstractmethod
    def GetUnit(self, index: int) -> "Unit":
        pass


class Interfaces(ABC):
    @abstractmethod
    def HasInterface(self, iid: IID) -> bool:
        pass

    @abstractmethod
    def FindInterface(self, iid: IID) -> "Interface":
        pass

    @abstractmethod
    def NumInterfaces(self) -> int:
        pass

    @abstractmethod
    def GetInterface(self, index: int) -> "Interface":
        pass


class Interface(ABC):
    @abstractmethod
    def GetInterfaceId(self) -> int:
        pass

    @abstractmethod
    def GetInterfaceVersion(self) -> int:
        pass

    @abstractmethod
    def GetObject(
        self,
    ) -> "BasicControlValue | ImageAcquisition | Properties | AutoCalibration":
        pass


class AutoCalibration(ABC):
    @abstractmethod
    def IsCalibrated(self) -> bool:
        pass

    @abstractmethod
    def Calibrate(self) -> None:
        pass


class BasicControlValue(ABC):
    @abstractmethod
    def MinControlValue(self) -> int:
        pass

    @abstractmethod
    def MaxControlValue(self) -> int:
        pass

    @abstractmethod
    def SetControlValue(self, pos: int) -> None:
        pass

    @abstractmethod
    def GetControlValue(self) -> int:
        pass

    @abstractmethod
    def GetMetricsConverters(self) -> "MetricsConverters":
        pass


class MetricsConverters(ABC):
    @abstractmethod
    def FindMetricsConverter(self, metricsId: EMetricsId) -> "MetricsConverter":
        pass


class MetricsConverter(ABC):
    @abstractmethod
    def GetMetricsValue(self, nControlValue: int) -> float:
        pass

    @abstractmethod
    def GetControlValue(self, dMetricsValue: float) -> int:
        pass

    @abstractmethod
    def GetMetricsId(self) -> int:
        pass


class Image(ABC):
    @abstractmethod
    def Format(self) -> "ImageFormat":
        pass

    @abstractmethod
    def LockPixelData(self) -> None:
        pass

    @abstractmethod
    def PixelData(self) -> "IntPtr":  # type: ignore # noqa
        pass

    @abstractmethod
    def UnlockPixelData(self) -> None:
        pass

    @abstractmethod
    def Dispose(self) -> None:
        pass


class ImageFormat(ABC):
    @abstractmethod
    def Width(self) -> int:
        pass

    @abstractmethod
    def Height(self) -> int:
        pass

    @abstractmethod
    def PixelFormat(self) -> PIXEL_FORMAT:
        pass

    @abstractmethod
    def Orientation(self) -> IMAGE_ORIENTATION:
        pass

    @abstractmethod
    def Stride(self) -> int:
        pass

    @abstractmethod
    def PixelBufferSize(self) -> int:
        pass


class ImageAcquisition(ABC):
    @abstractmethod
    def Acquire(
        self,
        context: "ImageAcquisitionContext",
        imageInfoPropIds: "IdList",  # type: ignore # noqa
    ) -> None:
        pass

    @abstractmethod
    def AcquireContinuous(
        self,
        context: "ImageAcquisitionContext",
        imageInfoPropIds: "IdList",  # type: ignore # noqa
    ) -> None:
        pass


class ImageAcquisitionContext(ABC):
    @abstractmethod
    def DoDelegateImageAcquired(
        self,
        delegateId: int,
        image: Image,
    ) -> None:
        pass

    @abstractmethod
    def DoDelegateAcquisitionEvent(
        self,
        delegateId: int,
        unit: Unit,
        acquisitionEventType: int,
    ) -> None:
        pass

    @property
    @abstractmethod
    def AcquisitionEventHandler(self) -> "DelegateAcquisitionEvent":
        pass

    @AcquisitionEventHandler.setter
    @abstractmethod
    def AcquisitionEventHandler(self, value: "DelegateAcquisitionEvent") -> None:
        pass

    @property
    @abstractmethod
    def ImageAcquiredHandler(self) -> "DelegateOnImageAcquired":
        pass

    @ImageAcquiredHandler.setter
    @abstractmethod
    def ImageAcquiredHandler(self, value: "DelegateOnImageAcquired") -> None:
        pass

    @property
    @abstractmethod
    def IsCancelled(self) -> bool:
        pass

    @IsCancelled.setter
    @abstractmethod
    def IsCancelled(self, value: bool) -> None:
        pass


class CancellableImageAcquisitionContext(ABC):
    @property
    @abstractmethod
    def SystemMemoryFactory(self) -> "CancellableImageAcquisitionContext":
        pass

    @property
    @abstractmethod
    def PropertyBagFactory(self) -> "CancellableImageAcquisitionContext":
        pass

    @abstractmethod
    def Dispose(self) -> None:
        pass

    @abstractmethod
    def DoDelegateImageAcquired(
        self,
        delegateId: int,
        image: Image,
    ) -> None:
        pass

    @abstractmethod
    def DoDelegateAcquisitionEvent(
        self,
        delegateId: int,
        unit: Unit,
        acquisitionEventType: int,
    ) -> None:
        pass

    @property
    @abstractmethod
    def AcquisitionEventHandler(self) -> "DelegateAcquisitionEvent":
        pass

    @AcquisitionEventHandler.setter
    @abstractmethod
    def AcquisitionEventHandler(self, value: "DelegateAcquisitionEvent") -> None:
        pass

    @property
    @abstractmethod
    def ImageAcquiredHandler(self) -> "DelegateOnImageAcquired":
        pass

    @ImageAcquiredHandler.setter
    @abstractmethod
    def ImageAcquiredHandler(self, value: "DelegateOnImageAcquired") -> None:
        pass

    @property
    @abstractmethod
    def IsCancelled(self) -> bool:
        pass

    @IsCancelled.setter
    @abstractmethod
    def IsCancelled(self, value: bool) -> None:
        pass


class DelegateAcquisitionEvent(ABC):
    @abstractmethod
    def Invoke(self, unit: Unit, imageAcquisitionEventType: int) -> None:
        pass


class DelegateOnImageAcquired(ABC):
    @abstractmethod
    def Invoke(self, image: Image) -> None:
        pass


class PropertyValue(ABC):
    @property
    @abstractmethod
    def UniqueId(self) -> int:
        pass

    @abstractmethod
    def GetValue(self):
        pass

    @abstractmethod
    def SetValue(self, value) -> None:
        pass


class Property(ABC):
    @abstractmethod
    def GetId(self) -> int:
        pass

    @abstractmethod
    def GetValue(self) -> "PropertyValue":
        pass

    @abstractmethod
    def GetInfo(self) -> "PropertyInfo":
        pass


class Properties(ABC):
    @abstractmethod
    def NumProperties(self) -> int:
        pass

    @abstractmethod
    def GetProperty(self, index: int) -> Property:
        pass

    @abstractmethod
    def FindProperty(self, propertyId: int) -> Property:
        pass


class PropertyInfo(ABC):
    @abstractmethod
    def GetAccessRights(self) -> int:
        pass

    @abstractmethod
    def GetDefaultValue(self) -> "PropertyValue":
        pass
