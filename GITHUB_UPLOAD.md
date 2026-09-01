# Private GitHub submission

The local repository is initialised on branch `main` and all release files are
staged. Set the real submitting author's Git identity before creating the first
commit; do not use a placeholder identity in the official history.

```bash
git config user.name "REAL AUTHOR NAME"
git config user.email "REAL AUTHOR EMAIL"
git commit -m "OASIS 2026 reproducible agent submission"
git tag -a oasis-2026-submission-v1 -m "Frozen OASIS submission"
```

Create an empty private GitHub repository, then run the commands GitHub shows,
for example:

```bash
git remote add origin git@github.com:ACCOUNT/luciana-oasis-geoai-agent.git
git push -u origin main
git push origin oasis-2026-submission-v1
```

In GitHub, open `Settings` -> `Collaborators` and invite
`jinmengrao@gmail.com`. Confirm that the invitation is accepted or still valid
before submitting the private repository URL.

No tracked file currently exceeds GitHub's 100 MB single-file limit. Git LFS is
therefore optional for this curated repository. The separate 1.5 GB full-data
archive should be shared through controlled storage rather than committed here.
