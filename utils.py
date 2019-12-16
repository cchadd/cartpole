from pathlib import Path
import base64
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
from IPython.display import clear_output


def show_video(directory):
    html = []
    for mp4 in Path(directory).glob("*.mp4"):
        print(mp4)
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay 
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
    
display = Display(visible=0, size=(1400, 900))
display.start()