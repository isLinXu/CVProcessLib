'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-06-30
@desc: 通过pybullet模拟机器狗行走及步态
'''
import pybullet as p
import time
import numpy as np


def setEnv():
    '''参数设置'''
    # 设置重力
    p.setGravity(0, 0, -9.81)
    # 设置摄像机角度
    p.setRealTimeSimulation(1)
    # 将相机以所需的角度和距离对准机器人
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-30, cameraPitch=-30,
                                 cameraTargetPosition=[0.0, 0.0, 0.25])


def createHill():
    '''创建背景'''
    # Scenery e.g. an inclined box
    boxHalfLength = 2.5
    boxHalfWidth = 2.5
    boxHalfHeight = 0.2
    sh_colBox = p.createCollisionShape(p.GEOM_BOX, halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight])
    mass = 1
    block = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                              basePosition=[-2, 4, -0.1], baseOrientation=[0.0, 0.1, 0.0, 1])


def createDogJoints():
    '''创建机器狗'''
    # 机身主体设置为红色，halfExtents设置长宽高
    base_body = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.45, 0.08, 0.02])
    # 赋予重量传感
    package_weight = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.08, 0.05])
    # two blue hinges on the side to which joints attach
    # 铰链
    roll_joint = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
    # 腿关节
    hip_joint = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
    # 膝关节
    knee_joint = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
    # 脚关节
    foot_joint = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)

    linkCollisionShapeIndices = [roll_joint, hip_joint, knee_joint, foot_joint,
                                 roll_joint, hip_joint, knee_joint, foot_joint,
                                 roll_joint, hip_joint, knee_joint, foot_joint,
                                 roll_joint, hip_joint, knee_joint, foot_joint,
                                 package_weight]
    return base_body, linkCollisionShapeIndices


def designJoints(botCenterToFront, botWidthfmCenter, xoffh, yoffh, botCenterToBack, upperLegLength, lowerLegLength):
    '''
    关节设计
    :param botCenterToFront:
    :param botWidthfmCenter:
    :param xoffh:
    :param yoffh:
    :param botCenterToBack:
    :param upperLegLength:
    :param lowerLegLength:
    :return:
    '''
    linkPositions = [[botCenterToFront, botWidthfmCenter, 0], [xoffh, yoffh, 0], [0, 0, -upperLegLength],
                     [0, 0, -lowerLegLength],
                     [botCenterToFront, -botWidthfmCenter, 0], [xoffh, -yoffh, 0], [0, 0, -upperLegLength],
                     [0, 0, -lowerLegLength],
                     [botCenterToBack, botWidthfmCenter, 0], [xoffh, yoffh, 0], [0, 0, -upperLegLength],
                     [0, 0, -lowerLegLength],
                     [botCenterToBack, -botWidthfmCenter, 0], [xoffh, -yoffh, 0], [0, 0, -upperLegLength],
                     [0, 0, -lowerLegLength],
                     [0, 0, +0.029]]

    # indices determine for each link which other link it is attached to
    # 通过索引确定每个链接所连接的其他链接
    # 例如,3rd index = 2表示左膝前关节与左前关节相连
    indices = [0, 1, 2, 3,
               0, 5, 6, 7,
               0, 9, 10, 11,
               0, 13, 14, 15,
               0]
    # 大多数关节是旋转的。棱柱关节暂时保持固定
    jointTypes = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_PRISMATIC,
                  p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_PRISMATIC,
                  p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_PRISMATIC,
                  p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_PRISMATIC,
                  p.JOINT_PRISMATIC]
    # 每个旋转关节的旋转轴
    axis = [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1],
            [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1],
            [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1],
            [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1],
            [0, 0, 1]]

    return linkPositions, indices, jointTypes, axis


# Function to calculate roll, hip and knee angles from the x,y,z coords of the foot wrt the hip.
## inverse kinematics
def xyztoang(x, y, z, yoffh, upperLegLength, lowerLegLength):
    '''
    逆运动学
    用于从脚相对于臀部的x、y、z坐标计算横滚、髋关节和膝关节角度。
    :param x:
    :param y:
    :param z:
    :param yoffh:
    :param upperLegLength:
    :param lowerLegLength:
    :return:
    '''
    dyz = np.sqrt(y ** 2 + z ** 2)
    lyz = np.sqrt(dyz ** 2 - yoffh ** 2)
    gamma_yz = -np.arctan(y / z)
    gamma_h_offset = -np.arctan(-yoffh / lyz)
    gamma = gamma_yz - gamma_h_offset

    lxzp = np.sqrt(lyz ** 2 + x ** 2)
    n = (lxzp ** 2 - lowerLegLength ** 2 - upperLegLength ** 2) / (2 * upperLegLength)
    beta = -np.arccos(n / lowerLegLength)

    alfa_xzp = -np.arctan(x / lyz)
    alfa_off = np.arccos((upperLegLength + n) / lxzp)
    alfa = alfa_xzp + alfa_off
    if any(np.isnan([gamma, alfa, beta])):
        print(x, y, z, yoffh, upperLegLength, lowerLegLength)
    return [gamma, alfa, beta]

def RotYawr(theta):
    '''
    仅在机器人框架和世界框架之间的偏航旋转矩阵
    计算：[cos(t),-sin(t),0,][sin(t),cos((t)),0][0,0,1]
    :param theta:
    :return:
    '''
    Rhor = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return Rhor


def setlegsxyz(xvec, yvec, zvec, eachMotorSpeed, botCenterToFront, botWidthfmCenter, yoffh, upperLegLength,
               lowerLegLength, dog, botCenterToBack):
    '''
    设置腿部axis
    :param xvec:
    :param yvec:
    :param zvec:
    :param eachMotorSpeed:
    :param botCenterToFront:
    :param botWidthfmCenter:
    :param yoffh:
    :param upperLegLength:
    :param lowerLegLength:
    :param dog:
    :param botCenterToBack:
    :return:
    '''
    a = xyztoang(xvec[0] - botCenterToFront, yvec[0] - botWidthfmCenter, zvec[0], yoffh, upperLegLength,
                 lowerLegLength)  # (x,y,z,yoffh,upperLegLength,lowerLegLength)
    spd = 1
    joint = 0
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=a[0], force=1000, maxVelocity=spd)
    joint = 1
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=a[1], force=1000,
                            maxVelocity=eachMotorSpeed[0])
    joint = 2
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=a[2], force=1000,
                            maxVelocity=eachMotorSpeed[0])

    a = xyztoang(xvec[1] - botCenterToFront, yvec[1] + botWidthfmCenter, zvec[1], -yoffh, upperLegLength,
                 lowerLegLength)  # (x,y,z,yoffh,upperLegLength,lowerLegLength)
    spd = 1.0
    joint = 4
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=a[0], force=1000, maxVelocity=spd)
    joint = 5
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=a[1], force=1000,
                            maxVelocity=eachMotorSpeed[1])
    joint = 6
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=a[2], force=1000,
                            maxVelocity=eachMotorSpeed[1])

    a = xyztoang(xvec[2] - botCenterToBack, yvec[2] - botWidthfmCenter, zvec[2], yoffh, upperLegLength,
                 lowerLegLength)  # (x,y,z,yoffh,upperLegLength,lowerLegLength)
    spd = 1.0
    joint = 8
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=a[0], force=1000, maxVelocity=spd)
    joint = 9
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=a[1], force=1000,
                            maxVelocity=eachMotorSpeed[2])
    joint = 10
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=a[2], force=1000,
                            maxVelocity=eachMotorSpeed[2])

    a = xyztoang(xvec[3] - botCenterToBack, yvec[3] + botWidthfmCenter, zvec[3], -yoffh, upperLegLength,
                 lowerLegLength)  # (x,y,z,yoffh,upperLegLength,lowerLegLength)
    spd = 1.0
    joint = 12
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=a[0], force=1000, maxVelocity=spd)
    joint = 13
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=a[1], force=1000,
                            maxVelocity=eachMotorSpeed[3])
    joint = 14
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=a[2], force=1000,
                            maxVelocity=eachMotorSpeed[3])


def dogBody(base_body, basePosition, baseOrientation, linkCollisionShapeIndices, linkPositions, indices, jointTypes,
            axis):
    '''
    机器狗的主体
    :param base_body:
    :param basePosition:
    :param baseOrientation:
    :param linkCollisionShapeIndices:
    :param linkPositions:
    :param indices:
    :param jointTypes:
    :param axis:
    :return:
    '''
    body_Mass = 1  ## assumption 1... we see that at 100 at carries well to certain distance. but ultimately falls
    visualShapeId = -1
    link_Masses = [.1, .1, .1, .1,  # roll to hip to knee to foot
                   .1, .1, .1, .1,
                   .1, .1, .1, .1,
                   .1, .1, .1, .1,
                   20]  ### 20 the weight of the payload
    nlnk = len(link_Masses)  ## number of links
    linkVisualShapeIndices = [-1] * nlnk  # =[-1,-1,-1, ... , -1]
    linkOrientations = [[0, 0, 0, 1]] * nlnk
    linkInertialFramePositions = [[0, 0, 0]] * nlnk
    # Note the orientations are given in quaternions (4 params). There are function to convert of Euler angles and back
    linkInertialFrameOrientations = [[0, 0, 0, 1]] * nlnk

    dog = p.createMultiBody(body_Mass, base_body, visualShapeId, basePosition, baseOrientation,
                            linkMasses=link_Masses,
                            linkCollisionShapeIndices=linkCollisionShapeIndices,
                            linkVisualShapeIndices=linkVisualShapeIndices,
                            linkPositions=linkPositions,
                            linkOrientations=linkOrientations,
                            linkInertialFramePositions=linkInertialFramePositions,
                            linkInertialFrameOrientations=linkInertialFrameOrientations,
                            linkParentIndices=indices,
                            linkJointTypes=jointTypes,
                            linkJointAxis=axis)
    return dog


def initialBalance(dog, joint):
    '''
    初始化平衡主体与关节
    :param dog:
    :param joint:
    :return:
    '''
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=0.01, force=1000, maxVelocity=3)
    # Same for the prismatic feet spheres
    joint = 3
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=0.0, force=1000, maxVelocity=3)
    joint = 7
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=0.0, force=1000, maxVelocity=3)
    joint = 11
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=0.0, force=1000, maxVelocity=3)
    joint = 15
    p.setJointMotorControl2(dog, joint, p.POSITION_CONTROL, targetPosition=0.0, force=1000, maxVelocity=3)


def addFrictionFootjoints(dog):
    '''为脚关节增加横向摩擦力'''
    # 用于第 3 关节、第 7 关节、第 11 关节、第 15 关节（所有接触关节）
    p.changeDynamics(dog, 3, lateralFriction=2)
    p.changeDynamics(dog, 7, lateralFriction=2)
    p.changeDynamics(dog, 11, lateralFriction=2)
    p.changeDynamics(dog, 15, lateralFriction=2)


def main():
    # 需要初始化 pybullet
    p.connect(p.GUI)
    p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(0, 0)
    #Camera paramers 能够偏航俯仰和缩放相机（焦点仍然在机器人上）
    cyaw = 10
    cpitch = -15
    cdist = 1.5
    # 创建一座小山
    createHill()
    # Create 创建一个由基本关节点组成的狗或加载一个urdf
    base_body, linkCollisionShapeIndices = createDogJoints()

    ## dog shape parameters
    ## shapes and weight defined.
    # link positions wrt the link they are attached to
    # distance of the COM from the front leg along x direction
    botCenterToFront = 0.4
    # distance of the COM from the back leg along -ve x direction
    botCenterToBack = -0.4
    # distance of the COM from the legs along y direction
    botWidthfmCenter = 0.1

    # 所有组合关节的定位
    xoffh = 0.05
    yoffh = 0.05
    upperLegLength = 0.3
    lowerLegLength = 0.3

    # 初始动力学参数
    yawInitial = 1.3  # 控制机器人的偏航

    linkPositions, indices, jointTypes, axis = designJoints(botCenterToFront, botWidthfmCenter, xoffh, yoffh,
                                                            botCenterToBack, upperLegLength, lowerLegLength)
    # 将身体放在场景中的以下身体坐标处
    basePosition = [0, 0, 1]
    baseOrientation = [0, 0, 0, 1]

    # 创建机器狗的主函数
    dog = dogBody(base_body, basePosition, baseOrientation, linkCollisionShapeIndices, linkPositions, indices,
                  jointTypes, axis)

    # Due to the weight the prismatic extraweight block needs to be motored up
    # 由于重量,棱柱形超重块需要被驱动起来
    initialBalance(dog, joint=16)

    # 设置重力
    setEnv()
    addFrictionFootjoints(dog)

    # 预初始化机器人位置
    setlegsxyz([botCenterToFront, botCenterToFront, botCenterToBack, botCenterToBack],
               [botWidthfmCenter + 0.1, -botWidthfmCenter - 0.1, botWidthfmCenter + 0.1, -botWidthfmCenter - 0.1],
               [-0.5, -0.5, -0.5, -0.5], [1, 1, 1, 1], botCenterToFront, botWidthfmCenter, yoffh, upperLegLength,
               lowerLegLength, dog, botCenterToBack)

    # 从初始配置行走状态暂停的时间
    t0 = time.time()
    t = time.time()
    while ((t - t0) < 4):
        t = time.time()

    # 将机器人中心设置为从原点1沿x和y的初始距离，沿Z为0.5
    robotCenter = np.array([1, 1, 0.5])  ### sets the init distance from the origin 1 along x and y, 0.5 alongz

    footJointsPtsI = np.array([[botCenterToFront, botCenterToFront, botCenterToBack, botCenterToBack],
                               [botWidthfmCenter + 0.1, -botWidthfmCenter - 0.1, botWidthfmCenter + 0.1,
                                -botWidthfmCenter - 0.1],
                               [-0.5, -0.5, -0.5, -0.5]])
    # 初始化机器狗的位置与方向
    quat = p.getQuaternionFromEuler([0, 0, yawInitial])
    # 将狗urdf的基本位置设置为机器人中心，将基本角度设置为quat
    p.resetBasePositionAndOrientation(dog, robotCenter,quat)
    # 初始化腿部绝对位置
    RyawInitial = RotYawr(yawInitial)
    # 将脚关节从机器人框架转换为世界框架，并通过robotCenter移动它们
    legsOnGroundPos = (np.dot(RyawInitial, footJointsPtsI).T + robotCenter).T  # Apply rotation plus translation

    yawRobotFrame = RotYawr(yawInitial)

    # 在机器人框架中重新调整腿部相对位置，并设置腿部
    newGroundPos = (legsOnGroundPos.T - robotCenter).T
    dlegsR = np.dot(yawRobotFrame.T, newGroundPos)
    setlegsxyz(dlegsR[0], dlegsR[1], dlegsR[2], [1, 1, 1, 1], botCenterToFront, botWidthfmCenter, yoffh, upperLegLength,
               lowerLegLength, dog, botCenterToBack)

    # 根据双脚位置的平均值计算一个新的机器人中心位置
    # 计算一个新的机器人偏航方向也从脚的位置
    frontLegsCenter = (legsOnGroundPos[:, 0] + legsOnGroundPos[:, 1]) / 2.0
    backLegCenter = (legsOnGroundPos[:, 2] + legsOnGroundPos[:, 3]) / 2.0
    fwdBwdCenterDistance = frontLegsCenter - backLegCenter

    # 步行速度（改变步行循环时间）
    walkLoopSpd = 400  # 400

    # 设置通用电机转速
    eachMotorSpeed = [12] * 4

    # 设置当前腿部位置
    currentLeg = 0
    # 将旋转的中心初始化为当前机器人位置
    centerOfRotation = backLegCenter

    # 将主体位置设置为机器人位置
    xoff = 0
    yoff = 0

    # 初始到步行前进
    dr = 0
    drp = 0

    # 行走步态设置
    #腿部顺序（对于旋转机器人，我选择按左前,右前,左后,右后的顺序切换腿部）
    lseq=[0,1,3,2]     # 步行步态
    lseqp=[0,1,3,2]    # 步行步态
    # lseq=[2,0,3,1]     # 小跑步态
    # lseqp=[2,0,3,1]    # 小跑步态
    # lseq = [0, 2, 1, 3]  # 缓行步态
    # lseqp = [0, 2, 1, 3] # 缓行步态
    # 设置按键控制
    while (1):
        cubePos, _ = p.getBasePositionAndOrientation(dog)
        p.resetDebugVisualizerCamera(cameraDistance=cdist, cameraYaw=cyaw, cameraPitch=cpitch,
                                     cameraTargetPosition=cubePos)

        keys = p.getKeyboardEvents()
        # 通过Keys来控制相机位置
        # 向右平移视角
        if keys.get(100):  # D
            cyaw += 1
        # 向左平移视角
        if keys.get(97):  # A
            cyaw -= 1
        # 向下调整视角
        if keys.get(99):  # C
            cpitch += 1
        # 向上调整视角
        if keys.get(102):  # F
            cpitch -= 1
        # 拉远距摄像头
        if keys.get(122):  # Z
            cdist += .01
        # 拉近摄像头
        if keys.get(120):  # X
            cdist -= .01
        #改变机器人行走的键（fwd, bkw, rot right, rot left）
        if keys.get(65297):  # Up 前
            drp = 0
        if keys.get(65298):  # Down 后
            drp = 2
        if keys.get(65296):  # Right 右
            drp = 1
            centerOfRotation = robotCenter  # Set the center for the robot rotation to the current robot pos
            # 改变腿的顺序，使前臂张开而不是闭合
            lseqp = [1, 0, 2, 3]
        if keys.get(65295):  # Left 左
            drp = 3
            centerOfRotation = robotCenter

            lseqp = [0, 1, 3, 2]

        # 时间循环
        timeCycle = int(((time.time() - t0) * walkLoopSpd) % 800)
        # 单腿运动200个单位。一个 800 个单位的 4 腿步行循环
        # 使用 <、>、%（模）和除法，我们可以轻松地在循环的特定部分执行某些操作
        # 仅在下一个循环开始时应用新的步行循环类型（例如从 fwd 到 bkw 的 chg）
        if timeCycle < 20 and (not dr == drp):
            dr = drp
            lseq = lseqp

        # 要移动的腿的索引
        currentLeg = int(timeCycle / 200)
        # 实际要移动的腿
        k = lseq[currentLeg]

        # 腿部循环开始时身体以机器人中心为中心
        # 然后逐渐向与要移动的腿相反的方向移动
        # 确保重心保持在其他3条腿上
        # 当动腿再次下降时身体中心回到机器人中心
        if int(timeCycle % 200) < 10:
            xoff = 0
            yoff = 0
        elif int(timeCycle % 200) < 80:
            xoff += 0.002 * (-1 + 2 * int(k / 2))
            yoff += 0.002 * (-1 + 2 * (k % 2))

        elif int(timeCycle % 200) > 160:
            xoff -= 0.004 * (-1 + 2 * int(k / 2))
            yoff -= 0.004 * (-1 + 2 * (k % 2))

        # Recalc leg rel pos in desired robot frame
        newGroundPos = (legsOnGroundPos.T - robotCenter).T  # Translate
        dlegsR = np.dot(yawRobotFrame.T, newGroundPos)  # Rotate (Note the inverse rotation is the transposed matrix)
        # Then apply the body movement and set the legs
        setlegsxyz(dlegsR[0] - xoff - 0.03, dlegsR[1] - yoff, dlegsR[2], eachMotorSpeed, botCenterToFront,
                   botWidthfmCenter, yoffh, upperLegLength, lowerLegLength, dog,
                   botCenterToBack)  # 0.03 is for tweaking the center of grav.

        if int(timeCycle % 200) > 80:
            newGroundPos = (legsOnGroundPos.T - centerOfRotation).T
            yawlO = np.arctan2(newGroundPos[1, k], newGroundPos[0, k])
            rlO = np.sqrt(newGroundPos[0, k] ** 2 + newGroundPos[1, k] ** 2)
            if dr == 0:
                legsOnGroundPos[0, k] = rlO * np.cos(yawlO) + centerOfRotation[0] + 0.01 * np.cos(yawInitial)
                legsOnGroundPos[1, k] = rlO * np.sin(yawlO) + centerOfRotation[1] + 0.01 * np.sin(yawInitial)
            elif dr == 1:
                yawlO -= 0.015
                legsOnGroundPos[0, k] = rlO * np.cos(yawlO) + centerOfRotation[0]
                legsOnGroundPos[1, k] = rlO * np.sin(yawlO) + centerOfRotation[1]
            elif dr == 2:
                legsOnGroundPos[0, k] = rlO * np.cos(yawlO) + centerOfRotation[0] - 0.01 * np.cos(yawInitial)
                legsOnGroundPos[1, k] = rlO * np.sin(yawlO) + centerOfRotation[1] - 0.01 * np.sin(yawInitial)
            elif dr == 3:
                yawlO += 0.015
                legsOnGroundPos[0, k] = rlO * np.cos(yawlO) + centerOfRotation[0]
                legsOnGroundPos[1, k] = rlO * np.sin(yawlO) + centerOfRotation[1]

            if int(timeCycle % 200) < 150:
                # 向上移动腿k
                legsOnGroundPos[2, k] += .006
            else:
                # #移动/保持所有腿到地面
                legsOnGroundPos[2, k] -= .006
        else:
            # 移动/保持所有腿到地面
            legsOnGroundPos[2, 0] = 0.0
            legsOnGroundPos[2, 1] = 0.0
            legsOnGroundPos[2, 2] = 0.0
            legsOnGroundPos[2, 3] = 0.0

        # 计算下一个循环的向量和矩阵
        xfrO = (legsOnGroundPos[:, 0] + legsOnGroundPos[:, 1]) / 2.0
        xbkO = (legsOnGroundPos[:, 2] + legsOnGroundPos[:, 3]) / 2.0
        robotCenter = (xfrO + xbkO) / 2.0
        robotCenter[2] = 0.5  # along z axis
        fwdBwdCenterDistance = xfrO - xbkO
        yawInitial = np.arctan2(fwdBwdCenterDistance[1], fwdBwdCenterDistance[0])
        yawRobotFrame = RotYawr(yawInitial)

        time.sleep(0.01)

    p.disconnect()

if __name__ == '__main__':
    main()
