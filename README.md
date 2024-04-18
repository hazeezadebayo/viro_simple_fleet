
Fleet Management Based on Postgresql DB.

# USAGE:
## Terminal 1:
```bash
$ cd docker_ws/env/dev1
$ export DISPLAY=:0.0
$ xhost +local:docker
$ docker-compose up --build
```

## Terminal 2:
```bash
$ docker exec -it dev1_tailscaled_1 bash
$ cd ros2_ws/nav2_ign && . /usr/share/gazebo/setup.sh; source install/setup.bash; export TURTLEBOT3_MODEL=waffle; ros2 launch viro_simple_fleet fleet_client.launch.py
```

## Terminal 3:
```bash
$ docker exec -it dev1_tailscaled_1 bash
$ cd ros2_ws/nav2_ign && . /usr/share/gazebo/setup.sh; source install/setup.bash; python3 viro_simple_fleet/scripts/fleet_mngr_main.py
```
