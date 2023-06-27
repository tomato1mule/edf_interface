from typing import TypeVar
from edf_interface.protocol import *

RobotStateType = TypeVar('RobotStateType')
PlanType = TypeVar('PlanType')
ObjectType = TypeVar('ObjectType')


from .demo_server import DashEdfDemoServer