# HumanRecorder test-data notice

HumanRecorder records semantic game decisions, player-visible game state, audit-only
engine state, run seed, timestamps, a random persistent actor ID, and the IDs, versions
and SHA-256 fingerprints of loaded Mod assemblies. It does not record Steam account IDs,
keyboard input, mouse input, screenshots, chat, or files outside its recording directory.

The random actor ID allows several runs from one volunteer to be grouped. Delete
`%LOCALAPPDATA%\SlayTheSpire2\HumanRecorder\actor_id.txt` while the game is closed to
generate a new identity. Review and send datasets only if you consent to sharing the
play schedule implied by timestamps and the installed-Mod fingerprint.
