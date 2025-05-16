# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 18:07:41 2025

@author: lukas
"""

import git
import numpy as np
import os
from datetime import date
import shutil
from classes import Cl_Manager



git_pull_switch = True


repo_path = r"C:\Users\Lukas\Amazon_Project\Text_Recognition"
if(git_pull_switch == True):
    
    repo = git.Repo(repo_path)

    #Pull latest changes
    origin = repo.remotes.origin
    origin.pull()

#Actual programm run
Manager = Cl_Manager()
repo_old_books = os.path.join(repo_path, "Processed_Books")
repo_new_books = os.path.join(repo_path, "New_Books")
repo_error_books = os.path.join(repo_path, "Error_Books")
if os.path.isdir(repo_old_books):
    old_books = [f for f in os.listdir(repo_old_books) if os.path.isdir(os.path.join(repo_old_books, f))]

if os.path.isdir(repo_new_books):
    new_books = [f for f in os.listdir(repo_new_books) if os.path.isdir(os.path.join(repo_new_books, f))]
    count_new = len(new_books)
    if(count_new != 0):
        #There are new books
        for i in range(count_new):
            #loop over books to process
            current_book = new_books[i]
            repo_book = os.path.join(repo_new_books, current_book)
            #check that book is not already processed
            if(current_book in old_books):
                print("book already processed")
                repo_error = os.path.join(repo_error_books, current_book)
                shutil.copytree(repo_book, repo_error)
                shutil.rmtree(repo_book)
            else:
                print("Book is new")
                new_book_folder_directory = os.path.join(repo_old_books,current_book)
                
                #start processing book
                Manager.process_new_book(current_book)
                print(repo_book)
                shutil.copytree(repo_book, new_book_folder_directory)
                shutil.rmtree(repo_book)
                
    
        
if(git_pull_switch==True):
    #Check for uncommited changes
    if repo.is_dirty(untracked_files=True):
        #Add all changes
        repo.git.add(A=True)

        #Commit changes
        today = date.today()
        repo.index.commit(f"{today}: Automation Pipeline")

        #Push to remote repository
        origin.push()
    else:
        print("There are no changes to commit")