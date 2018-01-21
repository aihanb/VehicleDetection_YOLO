# code based on:
#YAD2K https://github.com/allanzelener/YAD2K
#darkflow https://github.com/thtrieu/darkflow
#Darknet.keras https://github.com/sunshineatnoon/Darknet.keras

import numpy as np
import cv2

import matplotlib.pyplot as plt
import cv2
import os, glob
import imageio
imageio.plugins.ffmpeg.download()
import numpy as np
from moviepy.editor import VideoFileClip
import pickle

# get_ipython().magic('matplotlib inline')
# get_ipython().magic("config InlineBackend.figure_format = 'retina'")


# In[4]:
# global new_count 

new_count = 0



def show_images(images, cmap=None):
    cols = 2
    rows = (len(images)+1)//cols
    
    plt.figure(figsize=(10, 11))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        # use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape)==2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()


# In[5]:



test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]

show_images(test_images)


# In[6]:



# image is expected be in RGB color space
def select_rgb_white_yellow(image): 
    # white color mask
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

show_images(list(map(select_rgb_white_yellow, test_images)))


# In[7]:



def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

show_images(list(map(convert_hsv, test_images)))


# In[8]:


def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

show_images(list(map(convert_hls, test_images)))


# In[9]:



def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

white_yellow_images = list(map(select_white_yellow, test_images))

show_images(white_yellow_images)


# In[10]:



def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

gray_images = list(map(convert_gray_scale, white_yellow_images))

show_images(gray_images)


# In[11]:


def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


# In[12]:


blurred_images = list(map(lambda image: apply_smoothing(image), gray_images))

show_images(blurred_images)


# In[13]:


def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

edge_images = list(map(lambda image: detect_edges(image), blurred_images))

show_images(edge_images)


# In[14]:


def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

    
def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.95]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.6, rows*0.6] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)


# images showing the region of interest only
roi_images = list(map(select_region, edge_images))

show_images(roi_images)


# In[15]:


def hough_lines(image):
    """
    `image` should be the output of a Canny transform.
    
    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)


list_of_lines = list(map(hough_lines, roi_images))


# In[16]:


def draw_lines(image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
    # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
    if make_copy:
        image = np.copy(image) # don't want to modify the original
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


line_images = []
for image, lines in zip(test_images, list_of_lines):
    line_images.append(draw_lines(image, lines))
    
show_images(line_images)


# In[17]:


def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    # add more weight to longer lines    
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    
    return left_lane, right_lane # (slope, intercept), (slope, intercept)


# In[18]:


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))


# In[19]:


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    
    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.6         # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    
    return left_line, right_line

    
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)
             
    
lane_images = []
for image, lines in zip(test_images, list_of_lines):
    lane_images.append(draw_lane_lines(image, lane_lines(image, lines)))

    
show_images(lane_images)


# In[20]:

left_lane = []
right_lane = []


from collections import deque

QUEUE_LENGTH=50

class LaneDetector:
    def __init__(self):
        self.left_lines  = deque(maxlen=QUEUE_LENGTH)
        self.right_lines = deque(maxlen=QUEUE_LENGTH)

    def process(self, image):
        white_yellow = select_white_yellow(image)
        gray         = convert_gray_scale(white_yellow)
        smooth_gray  = apply_smoothing(gray)
        edges        = detect_edges(smooth_gray)
        regions      = select_region(edges)
        lines        = hough_lines(regions)
        left_line, right_line = lane_lines(image, lines)

        def mean_line(line, lines):
            if line is not None:
                lines.append(line)

            if len(lines)>0:
                line = np.mean(lines, axis=0, dtype=np.int32)
                line = tuple(map(tuple, line)) # make sure it's tuples not numpy array for cv2.line to work
            return line

        left_line  = mean_line(left_line,  self.left_lines)
        right_line = mean_line(right_line, self.right_lines)
        left_lane.append(left_line)
        right_lane.append(right_line)

        return draw_lane_lines(image, (left_line, right_line))


# In[21]:


def process_video(video_input, video_output):
    detector = LaneDetector()

    clip = VideoFileClip(os.path.join('', video_input))
    processed = clip.fl_image(detector.process)
    processed.write_videofile(os.path.join('LaneDetection_output_videos', video_output), audio=False)


# In[22]:

# process_video('project_video.mp4', 'laneDetection.mp4')

get_ipython().magic("time process_video('project_video.mp4', 'laneDetection.mp4')")

print(left_lane, right_lane)


PreviousBoxes = []
# count = 0

def load_weights(model,yolo_weight_file):
                
    data = np.fromfile(yolo_weight_file,np.float32)
    data=data[4:]
    
    index = 0
    for layer in model.layers:
        shape = [w.shape for w in layer.get_weights()]
        if shape != []:
            kshape,bshape = shape
            bia = data[index:index+np.prod(bshape)].reshape(bshape)
            index += np.prod(bshape)
            ker = data[index:index+np.prod(kshape)].reshape(kshape)
            index += np.prod(kshape)
            layer.set_weights([ker,bia])


class Box:

    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()
        self.label = int()


def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);


def yolo_net_out_to_car_boxes(net_out, threshold = 0.2, sqrt=1.8,C=20, B=2, S=7):
    class_num = 6
    boxes = []
    lst = []
    # global count
    global PreviousBoxes
    count = 0
    t = 0.05
    flag = 0


    # threshold = 2
    # Previous = []
    SS        =  S * S # number of grid cells
    prob_size = SS * C # class probabilities
    conf_size = SS * B # confidences for each grid cell
    
    probs = net_out[0 : prob_size]
    confs = net_out[prob_size : (prob_size + conf_size)]
    cords = net_out[(prob_size + conf_size) : ]
    probs = probs.reshape([SS, C])
    confs = confs.reshape([SS, B])
    cords = cords.reshape([SS, B, 4])
    
    for grid in range(SS):
        for b in range(B):
            bx   = Box()
            bx.c =  confs[grid, b]
            bx.x = (cords[grid, b, 0] + grid %  S) / S
            bx.y = (cords[grid, b, 1] + grid // S) / S
            bx.w =  cords[grid, b, 2] ** sqrt 
            bx.h =  cords[grid, b, 3] ** sqrt
            p = probs[grid, :] * bx.c
            
            if p[class_num] >= threshold:
                bx.prob = p[class_num]
                # if count not in lst:
                    # bx.label = count
                    # lst.append(count)
                # else:
                    # count +=1
                    
                boxes.append(bx)
                
    # combine boxes that are overlap
    boxes.sort(key=lambda b:b.prob,reverse=True)
    for i in range(len(boxes)):
        boxi = boxes[i]
        if boxi.prob == 0: continue
        for j in range(i + 1, len(boxes)):
            boxj = boxes[j]
            if box_iou(boxi, boxj) >= .4:
                boxes[j].prob = 0.
    boxes = [b for b in boxes if b.prob > 0.]

    # for i, b in enumerate(boxes):
    #     bmid_x = (b.x + b.w) / 2
    #     bmid_y = (b.y + b.h) / 2
    #     for i, pb in enumerate(PreviousBoxes):
    #         pmid_x = (pb.x + pb.w) / 2
    #         pmid_y = (pb.y + pb.h) / 2

    #         if((pmid_x + t > bmid_x) or (pmid_x - t < bmid_x)):
    #             if((pmid_y + t > bmid_y) or (pmid_y - t < bmid_y)):
    #                 b.label = pb.label
    #                 count += 1
    #                 flag = 1

    #         #oldCodeStart
    #         # if (pb.x + t > b.x):
    #         #     if (pb.y + t > b.y):
    #         #         if (pb.w + t < b.w):
    #         #             if (pb.h + t < b.h):
    #                         # b.label = pb.label
    #                         # count +=1
    #                         # flag = 1
    #                         #oldCode End
    #         # if (pb.x - t > b.x):
    #         #     if (pb.y - t > b.y):
    #         #         if (pb.w - t > b.w):
    #         #             if (pb.h - t > b.h):
    #         #                 b.label = pb.label
    #         #                 # count += 1
    #         #                 flag = 1
    #     if flag ==0:
    #         b.label = count
    #         count += 1
    #     flag = 0
    # PreviousBoxes = boxes

        # b.label = CalculateLabel(prev, curr)
    return boxes





def draw_box(boxes,im,crop_dim):
    lst = []
    # new_count += 1
    global PreviousBoxes
    global new_count
    count = 0
    threshold = 10
    flag = 0
    left_x1 = 0
    left_y1 = 0
    left_x2 = 0
    left_y2 = 0
    right_x1 = 0 
    right_y1 = 0
    right_x2 = 0
    right_y2 = 0
    pmid_y = 0
    pmid_x = 0
     # process_video('project_video.mp4', 'laneDetection.mp4')

    imgcv = im
    [xmin,xmax] = crop_dim[0]
    [ymin,ymax] = crop_dim[1]
    for b in boxes:
        h, w, _ = imgcv.shape
        left  = int ((b.x - b.w/2.) * w)
        right = int ((b.x + b.w/2.) * w)
        top   = int ((b.y - b.h/2.) * h)
        bot   = int ((b.y + b.h/2.) * h)
        left = int(left*(xmax-xmin)/w + xmin)
        right = int(right*(xmax-xmin)/w + xmin)
        top = int(top*(ymax-ymin)/h + ymin)
        bot = int(bot*(ymax-ymin)/h + ymin)
        
        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bot   > h - 1:   bot = h - 1
        thick = int((h + w) // 150)
        mid_y = (top + bot) / 2
        mid_x = (left + right) / 2
        for i, pb in enumerate(PreviousBoxes):
            pleft  = int ((pb.x - pb.w/2.) * w)
            pright = int ((pb.x + pb.w/2.) * w)
            ptop   = int ((pb.y - pb.h/2.) * h)
            pbot   = int ((pb.y + pb.h/2.) * h)
            pleft = int(pleft*(xmax-xmin)/w + xmin)
            pright = int(pright*(xmax-xmin)/w + xmin)
            ptop = int(ptop*(ymax-ymin)/h + ymin)
            pbot = int(pbot*(ymax-ymin)/h + ymin)
        
            if pleft  < 0    :  pleft = 0
            if pright > w - 1: pright = w - 1
            if ptop   < 0    :   ptop = 0
            if pbot   > h - 1:   pbot = h - 1
            pthick = int((h + w) // 150)

            pmid_y = (ptop + pbot) / 2
            pmid_x = (left + right) / 2

            if((pmid_x + threshold < mid_x) or (pmid_x - threshold > mid_x)):
                if((pmid_y + threshold < mid_y) or (pmid_y - threshold > mid_y)):
                    b.label = pb.label
                    count += 1
                    flag = 1
        if flag ==0:
            b.label = count
            count += 1
        flag = 0
        for i, t in enumerate(left_lane):
            if i == new_count:
                left_x1 = t[0][0]
                left_y1 = t[0][1]
                left_x2 = t[1][0]
                left_y2 = t[1][1] 
        for i, t in enumerate(right_lane):
            if i == new_count:
                right_x1 = t[0][0]
                right_y1 = t[0][1]
                right_x2 = t[1][0]
                right_y2 = t[1][1] 

        # print(new_count, ' = ', left_x1, left_y1, left_x2, left_y2, right_x1, right_y1, right_x2, right_y2)

        # apx_dist = round(((1 - (top - bot) / h)** 6),1)
        apx_dist = (1.7 * 2117.64) / ((bot - top))

        v1 = [left_x2-left_x1, left_y2-left_y1]   # Vector 1
        v2 = [left_x2-pmid_x, left_y2-pmid_y]   # Vector 1
        xL = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product

        v3 = [right_x2-right_x1, right_y2-right_y1]   # Vector 1
        v4 = [right_x2-pmid_x, right_y2-pmid_y]   # Vector 1
        xR = v3[0]*v4[1] - v3[1]*v4[0]  # Cross product
        if xL > 0 and xR > 0:
            cv2.rectangle(imgcv, (left, top), (right, bot), (255,0,0), thick)
            cv2.putText(imgcv , 'L - ' + 'C' + str(b.label) + ' - {}'.format(apx_dist), (left - 5, top - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), thickness=2)            
            # print ('on one side')
        elif xL < 0 and xR < 0:
            cv2.rectangle(imgcv, (left, top), (right, bot), (255,0,0), thick)
            cv2.putText(imgcv , 'R - ' + 'C' + str(b.label) + ' - {}'.format(apx_dist), (left - 5, top - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), thickness=2)            
            # print ('right')
        elif xL < 0 and xR > 0:
            cv2.rectangle(imgcv, (left, top), (right, bot), (255,0,0), thick)
            cv2.putText(imgcv , 'B - ' + 'C' + str(b.label) + ' - {}'.format(apx_dist), (left - 5, top - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), thickness=2)            
            # print ('right')
        else:
            cv2.rectangle(imgcv, (left, top), (right, bot), (255,0,0), thick)
            cv2.putText(imgcv , 'NA - ' + 'C' + str(b.label) + ' - {}'.format(apx_dist), (left - 5, top - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), thickness=2)            
            # print ('on the same line!')

        # apx_dist = round(((1 - (top - bot) / h)** 6),1)
        # x = is_between(x1, y1, x2, y2, a1, a2)
        # cv2.rectangle(imgcv, (left, top), (right, bot), (255,0,0), thick)
        # cv2.putText(imgcv , 'Car' + str(b.label) + ' - {}'.format(apx_dist), (left, top), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), thickness=2)

    PreviousBoxes = boxes
    
    new_count += 1
    # new_count = -1


    # for b in boxes:
    #     h, w, _ = imgcv.shape
    #     left  = int ((b.x - b.w/2.) * w)
    #     right = int ((b.x + b.w/2.) * w)
    #     top   = int ((b.y - b.h/2.) * h)
    #     bot   = int ((b.y + b.h/2.) * h)
    #     mid_y = (top + bot) / 2
    #     mid_x = (left + right) / 2 
    #     apx_dist = round(((1 - (top - bot))),1)
    #     cv2.putText(imgcv, '{}'.format(apx_dist), (int(mid_x),int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    return imgcv