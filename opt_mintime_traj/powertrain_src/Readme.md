# Powertrain Behavior
By switching on the powertrain behavior as explained in both `Readme.md` files on the top level of this repository and
in the directory `/opt_mintime_traj` you can consider the powertrain behavior (thermal behavior, power losses,
state of charge) during the trajectory optimization. This feature gets especially interesting when dealing with multiple
consecutive race laps, see main `Readme.md` `Step 6`.

The results can look like the following plot. It shows the temperatures of the
- electric machines <img src="https://latex.codecogs.com/gif.latex?T_\mathrm{Machine}" />,
- battery <img src="https://latex.codecogs.com/gif.latex?T_\mathrm{Battery}" />,
- inverters <img src="https://latex.codecogs.com/gif.latex?T_\mathrm{Inverter}" />,
- cooling liquid for the motor-inverter circuit <img src="https://latex.codecogs.com/gif.latex?T_\mathrm{Fluid_{MI}}" />,
- cooling liquid for the battery circuit <img src="https://latex.codecogs.com/gif.latex?T_\mathrm{Fluid}_{B}" />.

![Powertrain component temperatures whilst driving one race lap on the Berlin (Germany) Formel E track.](component_temperatures.PNG)

# References
Powertrain Behavior\
Herrmann, Passigato, Betz, Lienkamp\
Minimum Race-Time Planning-Strategy for an Autonomous Electric Racecar\
In Press, https://arxiv.org/abs/2005.07127 \
Contact person: [Thomas Herrmann](mailto:thomas.herrmann@tum.de).
