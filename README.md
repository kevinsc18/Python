# Python 
Basic Python

## Add Sample Files to Enhancement Python 

# new file add

# You can use below commands to add files from test2 repo to test repo as below:

### In local test repo
rm -rf test2
git clone https://github.com/eromoe/test2
git add test2/
git commit -am 'add files from test2 repo to test repo'
git push

###  
    git rm --cached your_folder_with_repo
    git commit -m "remove cached repo"
    git add your_folder_with_repo/
    git commit -m "Add folder"
    git push
####  ths is I use to push to remote
git push --set-upstream origin master