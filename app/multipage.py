"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

"""
This file is the framework for generating multiple Streamlit applications
through an object oriented framework.
"""

# Import necessary libraries
import streamlit as st

# Define the multipage class to manage the multiple apps in our program
class MultiPage:
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []

    def add_page(self, title, func) -> None:
        """Class Method to Add pages to the project
        Args:
            title ([str]): The title of page which we are adding to the list of apps

            func: Python function to render this page in Streamlit
        """

        self.pages.append({"title": title, "function": func})

    def run(self):
        # Drodown to select the page to run
        page = st.sidebar.selectbox(
            "Navigation", self.pages, format_func=lambda page: page["title"]
        )

        # run the app function
        page["function"]()
