# RDD2022 example-image attribution

These six examples are a minimal, unmodified sample from RDD2022 by Arya et al.:

| Repository filename | RDD2022 source filename | Intended example |
|---|---|---|
| `d00-longitudinal-crack.jpg` | `China_Drone_000166.jpg` | D00 |
| `d10-transverse-crack.jpg` | `United_States_000028.jpg` | D10 |
| `d20-alligator-crack.jpg` | `China_Drone_000162.jpg` | D20 |
| `d40-pothole.jpg` | `China_MotorBike_000195.jpg` | D40 |
| `mixed-damage.jpg` | `Japan_000215.jpg` | D10 + D20 |
| `no-annotated-damage.jpg` | `Czech_000001.jpg` | empty label |

Source and citation: <https://doi.org/10.6084/m9.figshare.21431547>. The transport
mirror used for the experiment and the dataset maintainer repository identify the
dataset license as CC BY-SA 4.0. These image files remain available under CC BY-SA
4.0; the surrounding RoadGuard source code is AGPL-3.0-only. Labels in the table
describe the source annotations, not guaranteed model predictions. The application
runs the production checkpoint when an example is selected; no boxes are hardcoded.
