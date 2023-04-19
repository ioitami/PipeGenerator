# PipeGenerator

50.017 Graphics & Visualisation
Term 6 Jan-Apr - Professor Peng Song
Team 1: 
| Leong Keng Hoy (1005164) 
| Kim Si Eun (1005370) 
| Jowie Ng (1005494) 
| Sharryl Seto (1005523)

## Folder contents
|- Blender 3.4
    	|- blender (application)
	|- blender-launcher
|- src
	|- PipeGenerator.py
	|- PipeGeneration.blend
	|- lib 
|- screenshots
|- README.txt
|- 50.017_team1_slides.pdf
|- 50.017_team1_report.pdf
|- 50.017_team1_video.mp4

Watch our demo video here: https://youtu.be/x9o-llCBsCc

## How to run
1. Run the Blender (v3.4.1) application.
2. Open the PipeGeneration.blend file.
3. After the file loads, open the 'Text Editor' window and load the PipeGenerator.py file (if it isn't open).
	- Click "Open" (folder button) and select the PipeGenerator.py file. 
5. Ensure that the script is loaded. Reload by pressing Alt+R in the text editor.
6. Select an object. 
    	- Create a new mesh if you have no objects yet.
7. Click the "play button" to run the Python script.
Note: the script will give an error if you try to run it without selecting an object.

## How to use the interface
1. After running the Python script, a pop-up UI will appear in the '3D Viewport' window.
    	- It is named "PipeGenerator", with the options "Add Pipes" and "Delete Object Pipes".
2. Click "Add Pipes". Another UI panel called "Add Pipes" will appear on the bottom left of the '3D viewport' window.
    	- Here, you can control the pipes generated. By default, the Python script algorithm will be used.
3. To change the algorithm used, click the first dropdown button and select accordingly.
    	- Please note that for Geometry Node (GeoNode) algorithms, the UI to control the variables is in the 'Properties' window on the right
    	- In the 'Properties' window, click 'Modifier Properties' (wrench icon) to view the UI.

Currently, "Delete Object Pipes" only works for the Python script.

## Troubleshooting
1. You may come across this error when running the Python script: ModuleNotFoundError: No module named 'networkx'.
Reason: This occurs if the current .blend file open is not in the same folder as the .py file.
Fix: If you are working on a new .blend file, please save it to the src folder before running the Python script.

2. If Blender is unable to open, please redownload the portable version here: https://mirror.freedif.org/blender/release/Blender3.4/


Future work includes bug fixing and allowing the user to choose their own materials from their .blend files for the pipes.

Thank you and have fun!