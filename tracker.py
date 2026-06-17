import math
class Tracker:

    def __init__(self,max_distance=70):
        self.objects={}
        self.id=0
        self.max_dis=max_distance
        self.prev_positions={}

    def update(self,detections):
        results=[]
        #results.append["xmin","ymin","xmax","ymax","class id"]
        for(xmin,ymin,xmax,ymax)in detections:
            cx=(xmin+xmax)//2
            cy=(ymin+ymax)//2
        
            assigned_id=None
            min_dist=self.max_dis
            for obj_id,(px,py) in self.objects.items():
                dist=math.hypot(cx-px,cy-py)
                if dist<min_dist:
                    min_dist=dist
                    assigned_id=obj_id
                    self.prev_positions[assigned_id]=(px,py)
            
            if assigned_id==None:
                assigned_id=self.id
                self.prev_positions[assigned_id]=(cx,cy)
                self.id+=1

            self.objects[assigned_id]=(cx,cy)
            results.append([xmin,ymin,xmax,ymax,assigned_id])
            
        return results