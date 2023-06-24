from typing import TypeVar

RobotStateType = TypeVar('RobotStateType')
PlanType = TypeVar('PlanType')
ObjectType = TypeVar('ObjectType')


PLAN_FAIL = 'PLAN_FAIL'
EXECUTION_FAIL = 'EXECUTION_FAIL'
SUCCESS = 'SUCCESS'
RESET = 'RESET'
FEASIBLE = 'FEASIBLE'
INFEASIBLE = 'INFEASIBLE'
TERMINATE = 'TERMINATE'

from .demo_server import DashEdfDemoServer