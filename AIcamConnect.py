import ntcore


inst = ntcore.NetworkTableInstance.getDefault()


inst.startClient4("Vision Game Pieces")

inst.setServer("10.45.72.2")


table = inst.getTable("Vision")

while True:
    
    
    
    
    table.putNumber("Camera1", (x, y, z, d))
    table.putNumber("Camera2", (x, y, z, d))
    table.putNumber("Camera3", (x, y, z, d))
    table.putNumber("Camera4", (x, y, z, d))
    # get all four cameras
    

