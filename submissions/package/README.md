# Final submission package (split for GitHub's 100 MB file limit)

Rejoin the parts into one zip, then rename it `<team_name>_submission.zip` and upload:

Windows (cmd):
    copy /b TEAM_submission.zip.001+TEAM_submission.zip.002+TEAM_submission.zip.003+TEAM_submission.zip.004 TEAM_submission.zip

PowerShell:
    cmd /c "copy /b TEAM_submission.zip.001+TEAM_submission.zip.002+TEAM_submission.zip.003+TEAM_submission.zip.004 TEAM_submission.zip"

mac/linux:
    cat TEAM_submission.zip.* > TEAM_submission.zip

Contents (challenge layout): output/matching_results.tsv (the 0.933 leaderboard file) +
output/candidate_pairs.tsv, code/business_entity_resolution/ (src, scripts, models, README,
requirements.txt: the exact code that produced those outputs), Documentation_template.md.
Before uploading, open Documentation_template.md and put the team name on the "Team Name" line.
