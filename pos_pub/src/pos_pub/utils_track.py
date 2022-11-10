import cv2
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('agg')
import pyrealsense2 as rs
import scipy.optimize
import functools
from skspatial.objects import Plane
from skspatial.objects import Point
import vg
from pytransform3d.rotations import matrix_from_axis_angle
from scipy.spatial.transform import Rotation as Rot

def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a*x + b*y + c
    return z

def error(params, points):
    result = 0
    for (x,y,z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff**2
    return result

def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]

def get_rotxyz(vector, normal, points):
    """
    Rotates a 3xn array of 3D coordinates from the +z normal to an
    arbitrary new normal vector.
    """
    points = np.array(points)
    vector = vg.normalize(vector)
    axis = vg.perpendicular(normal, vector)
    angle = vg.angle(normal, vector, units='rad')
    axis_angle = np.hstack((axis, (angle,)))
    R = matrix_from_axis_angle(axis_angle)
    r = Rot.from_matrix(R)
    rot_points = r.apply(points)
    return rot_points

def get_tps_u(Pt_a, Pt_b):
    D = np.sqrt(np.square(Pt_a[:, None, :2] - Pt_b[None, :, :2]).sum(-1))
    U = D**2 * np.log(D+1e-6)
    return U

def get_tps_coef(control, target, lmda = 0):
    n = control.shape[0]
    U = get_tps_u(control, control)
    K = U + np.eye(n, dtype=np.float32) * lmda
    P = np.ones((n,3), dtype=np.float32)
    V = np.zeros((n+3,2),dtype=np.float32)
    P[:,1:] = control
    V[:n,:2] = target
    L = np.zeros((n+3,n+3), dtype=np.float32)
    L[:n,:n] = K
    L[:n,-3:] = P
    L[-3:,:n] = P.T
    tps_coef = np.linalg.solve(L, V)
    return tps_coef

def tps_trans(source, control, coef):
    n = source.shape[0]
    U = get_tps_u(source, control)
    L = np.hstack([U,np.ones((n,1)),source])
    trans_after = np.dot(L, coef)
    return trans_after

def get_tps_mat(VERBOSE=True):
    pipeline = rs.pipeline()
    config = rs.config()
    align_to = rs.stream.color
    align = rs.align(align_to)
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)

    x = np.linspace(170,470,4)
    y = np.linspace(90,390,4)
    X, Y = np.meshgrid(x,y)
    ctrl_xy = np.stack([X,Y],axis=2).reshape(-1,2)
    real_pt = []

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # pipeline
    pipeline.start(config)
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        for i,j in enumerate(ctrl_xy):
            pt = (int(ctrl_xy[i,0]),int(ctrl_xy[i,1]))
            cv2.line(images, (pt),(pt),(0,0,255),5)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        if cv2.waitKey(20) == 27:
            for i,j in enumerate(ctrl_xy):
                pt = (int(ctrl_xy[i,0]),int(ctrl_xy[i,1]))
                depth = depth_frame.get_distance(int(ctrl_xy[i,0]),int(ctrl_xy[i,1]))
                x,y,z = rs.rs2_deproject_pixel_to_point(depth_intrin,pixel=[int(ctrl_xy[i,0]),int(ctrl_xy[i,1])],depth=depth)
                real_pt.append([x,y,z])
            break

    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()

    # Do plane regression
    real_x, real_y, real_z = zip(*real_pt)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(real_x, real_y, real_z)

    z_errors = functools.partial(error, points=real_pt)
    plane_params = [0, 0, 0]
    final_params = scipy.optimize.minimize(z_errors, plane_params)

    a = final_params.x[0]
    b = final_params.x[1]
    c = final_params.x[2]

    point  = np.array([0.0, 0.0, c])
    normal = np.array(cross([1,0,a], [0,1,b]))
    d = -point.dot(normal)
    plane_x, plane_y = np.meshgrid([-2,2], [-2,2])
    plane_z = (-normal[0] * plane_x - normal[1] * plane_y - d) * 1. /normal[2]
    ax.plot_surface(plane_x, plane_y, plane_z, alpha=0.2, color=[0,1,0])

    plane = Plane(point=point, normal=normal)
    projected_ls = []

    for i in real_pt:
        point = Point(i)
        point_projected = plane.project_point(point)
        # vector_projection = Vector.from_points(point, point_projected)
        projected_ls.append(point_projected)

    proj_x, proj_y, proj_z = zip(*projected_ls)
    ax.scatter(proj_x, proj_y, proj_z, c='r')

    vector_z = np.array([0,0,-1])
    RotXYZ = get_rotxyz(vector_z, normal, projected_ls)
    RotXYZ[:,2] = 0

    rot_x, rot_y, rot_z = zip(*RotXYZ)
    ax.scatter(rot_x, rot_y, rot_z, c='g')
    plt.show()

    target_xy = np.zeros_like(ctrl_xy)
    target_xy[:,:] = RotXYZ[:,0:2]

    tps_coef = get_tps_coef(ctrl_xy,target_xy)

    input_pts = np.linspace(100, 400, 15)
    input_pt = np.zeros((input_pts.shape[0],2))
    input_pt [:,0] = 340
    input_pt [:,1] = input_pts
    output_pt = tps_trans(input_pt, ctrl_xy, tps_coef)

    # if VERBOSE:
    #     plt.figure(figsize=(10,5))
    #     plt.subplot(1,2,1)
    #     plt.scatter(ctrl_xy[:,0],ctrl_xy[:,1],c="b")
    #     plt.scatter(input_pt[:,0],input_pt[:,1],c="r")
    #     ax2 = plt.subplot(1,2,2)
    #     plt.scatter(target_xy[:,0],target_xy[:,1],c="b")
    #     plt.scatter(output_pt[:,0],output_pt[:,1],c="r")
    #     ax2.set_xlim([-1, 1])
    #     ax2.set_ylim([0.5, 4])
    #     plt.show()

    return tps_coef

if __name__ == "__main__":
    get_tps_mat()