# Publish Milestone 01 on GitHub

1. Obtain approval from your supervising professor for public sharing of the project name, code, approach, images and preliminary research logs. Do not expose unpublished research or unlicensed dataset images.
2. Unzip the repository. Read `README.md` and `RESEARCH_LOG.md` and confirm they match your actual work; attribute code assistance appropriately under your institution's AI policy.
3. Install and run the **pretrained** experiment on a machine with network access and `pip install -r requirements.txt`:

   ```bash
   python -m src.run_experiment --weights pretrained
   python -m pytest -q
   ```

4. Check the generated `results/pretrained_resnet18/metrics.json` and `embeddings.npz`; confirm the 512-D features and repeatability. These are technical observations, **not** Nepali-caption results.
5. Update `RESEARCH_LOG.md` with the run date, actual numbers, runtime, input-image sources, what you personally did, limitations and next questions. Distinguish supervisor-designed parts from your own contributions.
6. Create a public GitHub repository named `nepali-image-captioning-reformer` only after approval. Upload the folder contents (not a nested extra folder), commit with title `Milestone 01: reproducible ResNet18 image feature extraction`, and keep the README as the repository front page. You may alternatively use Git command line:

   ```bash
   git init
   git add .
   git commit -m "Milestone 01: reproducible ResNet18 image feature extraction"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/nepali-image-captioning-reformer.git
   git push -u origin main
   ```

   The remote repository must already exist; replace `YOUR_USERNAME`. Git may require identity configuration and an authenticated connection. Uploads through GitHub's website are also acceptable.
7. Link the GitHub repository from your CV under **Ongoing Research — Nepali Image Caption Generation Using Reformer (Milestone 01: Image Encoder Experiment)**, without suggesting that the Reformer model or a captioning dataset is implemented.

*No GitHub publication or pretrained run is completed by merely receiving this ZIP.*
