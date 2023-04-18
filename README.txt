# PipeGenerator

50.017 Graphics & Visualisation
Term 6 Jan-Apr - Professor Peng Song
Team 1: 
| Leong Keng Hoy (1005164) 
| Kim Si Eun (1005370) 
| Jowie Ng (1005494) 
| Sharryl Seto (1005523)

## Folder contents
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
3. After the file loads, open the 'Text Editor' window.
4. Click "Open" (folder button) and select the PipeGenerator.py file. 
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

Future work includes bug fixing and allowing the user to choose their own materials from their .blend files for the pipes.

Thank you and have fun!