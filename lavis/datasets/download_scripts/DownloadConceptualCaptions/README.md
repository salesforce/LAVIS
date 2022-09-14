<!--
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
-->

# Download Conceptual Captions Data

Place data from: https://ai.google.com/research/ConceptualCaptions/download in this folder

`Train_GCC-training.tsv / cc3m.tsv` Training Split (3,318,333)

run `download_data_cc3m.py` or `download_data_cc12m.py`.

Images will be in default LAVIS cache folders. You can stop and resume, the settings for splitting downloads into chunks / threads are not optimal, but it maxed out my connection so i kept them as is.

Note: A previous version of this script used a different file naming scheme, this changed and if you are resuming a previously started download, you will get duplicates.

A bunch of them will fail to download, and return web pages instead. These will need to be cleaned up later. See `downloaded_validation_report.tsv` after it downloads for HTTP errors. Around 8% of images are gone, based on validation set results. Setting the user agent could fix some errors too maybe - not sure if any requests are rejected by sites based on this.

It should take about a day or two to download the training data, keep an eye on disk space.
