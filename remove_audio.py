#%%
from moviepy.editor import VideoFileClip

videoclip = VideoFileClip("static/videos/trial4_trim.mov")
new_clip = videoclip.without_audio()
new_clip.write_videofile("static/videos/trial4_trim.mov", codec="libx264", audio_codec="aac")
# %%
