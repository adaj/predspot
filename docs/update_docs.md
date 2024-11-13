```
cd docs
make clean
make html
cd ..

# Delete local gh-pages branch if it exists
git branch -D gh-pages

# Delete remote gh-pages branch
git push origin --delete gh-pages

# Create new gh-pages branch without history
git checkout --orphan gh-pages

# Copy the built files to root
cp -r docs/build/html/* .

# Add and commit
git add .
git commit -m "Deploy documentation"

# Force push the new branch
git push -f origin gh-pages

# Go back to your previous branch (probably main or master)
git checkout -
```