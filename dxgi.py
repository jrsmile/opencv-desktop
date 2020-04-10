import d3dshot
import time
import numpy as np
import cv2
from PIL import Image
import glob

templates=[]
template_name=[]
threshold = 0.75
scale_percent = 100
region=(1050, 1900, 2800, 2159)
color = (0, 255, 0) 

filelist = glob.glob('templates-druid/*.blp')
print("Found {} icons, loading to ram...".format(len(filelist)))

for filename in filelist:
    looped_image = Image.open(filename) #load image
    looped_image_np = np.asarray(looped_image) #covert to numpy array
    width = int(looped_image_np.shape[1] * scale_percent / 100)
    height = int(looped_image_np.shape[0] * scale_percent / 100)
    dim = (width, height)
    looped_image_resized = cv2.resize(looped_image_np, dim, interpolation = cv2.INTER_AREA) #resize
    looped_image_cv = cv2.cvtColor(looped_image_resized, cv2.COLOR_RGB2BGR) # convert to grayscale
    #templates.append(looped_image_cv)
    templates.append(looped_image_cv[10:45,10:50].copy()) # crop upper left quater 
    template_name.append(filename) # save filename to list
    looped_image.close() # close original image
    print(".", end = '')

print(template_name)
d = d3dshot.create(capture_output="numpy")
#d.display = d.displays[0] # primary is default
d.capture(region = region)
print('Waiting 3 Seconds for first Frame...')
time.sleep(3)
print('Starting Main Loop. Stop with Ctrl+C')
fps = 0
tic = time.perf_counter()
try:
    while True:
        found = 0
        try:
            img= d.get_latest_frame()
        except Exception as Err:
            print('Error: ')
        
        #print("New Frame")
         # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        preview = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        #resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        #edges = cv2.Canny(resized,100,200)

        for template in templates: 
            res = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where( res >= threshold)
            if loc[0].size > 0:
                for pt in zip(*loc[::-1]):
                    cv2.rectangle( preview, pt , (pt[0] + template.shape[0], pt[1] + template.shape[1]) , color, 1 )
                
                found += 1
        
        toc = time.perf_counter()
        fps += 1
        if toc - tic >= 1:
            print(f"loop took {toc - tic:0.4f} seconds({fps} FPS), {found} spells found.")
            found = 0
            fps = 0
            tic = toc
            
            cv2.imshow("Preview", preview)

        if cv2.waitKey(1) == 27: # ESC is pressed
            break
except KeyboardInterrupt:
    print('interrupted!')
d.stop()
