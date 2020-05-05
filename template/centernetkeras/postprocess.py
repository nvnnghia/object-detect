from PIL import Image, ImageDraw
import numpy as np 

def NMS_all(predicts,category_n, pred_out_h, pred_out_w, score_thresh,iou_thresh):
  y_c=predicts[...,category_n]+np.arange(pred_out_h).reshape(-1,1)
  x_c=predicts[...,category_n+1]+np.arange(pred_out_w).reshape(1,-1)
  height=predicts[...,category_n+2]*pred_out_h
  width=predicts[...,category_n+3]*pred_out_w

  count=0
  for category in range(category_n):
    predict=predicts[...,category]
    mask=(predict>score_thresh)
    #print("box_num",np.sum(mask))
    if mask.all==False:
      continue
    box_and_score=NMS(predict[mask],y_c[mask],x_c[mask],height[mask],width[mask],iou_thresh,pred_out_h, pred_out_w)
    box_and_score=np.insert(box_and_score,0,category,axis=1)#category,score,top,left,bottom,right
    if count==0:
      box_and_score_all=box_and_score
    else:
      box_and_score_all=np.concatenate((box_and_score_all,box_and_score),axis=0)
    count+=1
  score_sort=np.argsort(box_and_score_all[:,1])[::-1]
  box_and_score_all=box_and_score_all[score_sort]
  #print(box_and_score_all)

 
  _,unique_idx=np.unique(box_and_score_all[:,2],return_index=True)
  #print(unique_idx)
  return box_and_score_all[sorted(unique_idx)]
  
def NMS(score,y_c,x_c,height,width,iou_thresh,pred_out_h, pred_out_w,merge_mode=False):
  if merge_mode:
    score=score
    top=y_c
    left=x_c
    bottom=height
    right=width
  else:
    #flatten
    score=score.reshape(-1)
    y_c=y_c.reshape(-1)
    x_c=x_c.reshape(-1)
    height=height.reshape(-1)
    width=width.reshape(-1)
    size=height*width
    
    
    top=y_c-height/2
    left=x_c-width/2
    bottom=y_c+height/2
    right=x_c+width/2
    
    inside_pic=(top>0)*(left>0)*(bottom<pred_out_h)*(right<pred_out_w)
    outside_pic=len(inside_pic)-np.sum(inside_pic)
    #if outside_pic>0:
    #  print("{} boxes are out of picture".format(outside_pic))
    normal_size=(size<(np.mean(size)*10))*(size>(np.mean(size)/10))
    score=score[inside_pic*normal_size]
    top=top[inside_pic*normal_size]
    left=left[inside_pic*normal_size]
    bottom=bottom[inside_pic*normal_size]
    right=right[inside_pic*normal_size]
  

    

  #sort  
  score_sort=np.argsort(score)[::-1]
  score=score[score_sort]  
  top=top[score_sort]
  left=left[score_sort]
  bottom=bottom[score_sort]
  right=right[score_sort]
  
  area=((bottom-top)*(right-left))
  
  boxes=np.concatenate((score.reshape(-1,1),top.reshape(-1,1),left.reshape(-1,1),bottom.reshape(-1,1),right.reshape(-1,1)),axis=1)
  
  box_idx=np.arange(len(top))
  alive_box=[]
  while len(box_idx)>0:
  
    alive_box.append(box_idx[0])
    
    y1=np.maximum(top[0],top)
    x1=np.maximum(left[0],left)
    y2=np.minimum(bottom[0],bottom)
    x2=np.minimum(right[0],right)
    
    cross_h=np.maximum(0,y2-y1)
    cross_w=np.maximum(0,x2-x1)
    still_alive=(((cross_h*cross_w)/area[0])<iou_thresh)
    if np.sum(still_alive)==len(box_idx):
      print("error")
      print(np.max((cross_h*cross_w)),area[0])
    top=top[still_alive]
    left=left[still_alive]
    bottom=bottom[still_alive]
    right=right[still_alive]
    area=area[still_alive]
    box_idx=box_idx[still_alive]
  return boxes[alive_box]#score,top,left,bottom,right



def draw_rectangle(box_and_score,img,color):
  number_of_rect=np.minimum(500,len(box_and_score))
  
  for i in reversed(list(range(number_of_rect))):
    top, left, bottom, right = box_and_score[i,:]

    
    top = np.floor(top + 0.5).astype('int32')
    left = np.floor(left + 0.5).astype('int32')
    bottom = np.floor(bottom + 0.5).astype('int32')
    right = np.floor(right + 0.5).astype('int32')
    #label = '{} {:.2f}'.format(predicted_class, score)
    #print(label)
    #rectangle=np.array([[left,top],[left,bottom],[right,bottom],[right,top]])

    draw = ImageDraw.Draw(img)
    #label_size = draw.textsize(label)
    #print(label_size)
    
    #if top - label_size[1] >= 0:
    #  text_origin = np.array([left, top - label_size[1]])
    #else:
    #  text_origin = np.array([left, top + 1])
    
    thickness=4
    if color=="red":
      rect_color=(255, 0, 0)
    elif color=="blue":
      rect_color=(0, 0, 255)
    else:
      rect_color=(0, 0, 0)
      
    
    if i==0:
      thickness=4
    for j in range(2*thickness):#薄いから何重にか描く
      draw.rectangle([left + j, top + j, right - j, bottom - j],
                    outline=rect_color)
    #draw.rectangle(
    #            [tuple(text_origin), tuple(text_origin + label_size)],
    #            fill=(0, 0, 255))
    #draw.text(text_origin, label, fill=(0, 0, 0))
    
  del draw
  return img
            
  
def check_iou_score(true_boxes,detected_boxes,iou_thresh):
  iou_all=[]
  for detected_box in detected_boxes:
    y1=np.maximum(detected_box[0],true_boxes[:,0])
    x1=np.maximum(detected_box[1],true_boxes[:,1])
    y2=np.minimum(detected_box[2],true_boxes[:,2])
    x2=np.minimum(detected_box[3],true_boxes[:,3])
    
    cross_section=np.maximum(0,y2-y1)*np.maximum(0,x2-x1)
    all_area=(detected_box[2]-detected_box[0])*(detected_box[3]-detected_box[1])+(true_boxes[:,2]-true_boxes[:,0])*(true_boxes[:,3]-true_boxes[:,1])
    iou=np.max(cross_section/(all_area-cross_section))
    #argmax=np.argmax(cross_section/(all_area-cross_section))
    iou_all.append(iou)
  score=2*np.sum(iou_all)/(len(detected_boxes)+len(true_boxes))
  print("score:{}".format(np.round(score,3)))
  return score
