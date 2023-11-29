set -e
# The following script has been modified from https://raw.githubusercontent.com/treeder/bump/master/gitbump.sh 

# Parse arguments
if [ $# -eq 0 ]; then
    echo "No command specified."
    exit 1
elif [ $# -gt 1 ]; then
    echo "Too many arguments specified."
    exit 2
else
    if [ $1 == "major" ];then
        bump_cmd="major"
    elif [ $1 == "minor" ];then
        bump_cmd="minor"
    elif [ $1 == "patch" ];then
        bump_cmd="patch"
    else
        echo "Invalid command given as argument: "$1""
        exit 3
    fi
fi

echo "Type of version bump: "$bump_cmd

git fetch --tags # checkout action does not get these

# git describe has issues with GitHub Actions: https://github.com/treeder/firetils/commit/160ef4560d8855c9c05f4cae207baeb71b7791f3/checks?check_suite_id=414542684
# oldv=$(git describe --match "v[0-9]*" --abbrev=0 HEAD)
# This new way seems to work better and avoids the issue above:
# -v:refname is a version sort
oldv=$(git tag --sort=-v:refname --list "v[0-9]*" | head -n 1)
echo "Old version: "$oldv

# if there is no version tag yet, let's start at 0.0.0
if [ -z "$oldv" ]; then
   echo "No existing version, starting at 0.0.0"
   oldv="0.0.0"
fi

newv=$(docker run --rm -v "$PWD":/app treeder/bump --input "$oldv" "$bump_cmd")
echo "New versiob: "$newv

git tag -a "v$newv" -m "version $newv"
git push --follow-tags
