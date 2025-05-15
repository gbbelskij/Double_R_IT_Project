def assign_user_tags(user_answers):
    tags = {
        "Lang": [],
        "Department": [],
        "Add tags": [],
        "Lv1": []
    }

    if 0 in user_answers:
        answer = user_answers[0]
        if answer == 0: 
            if 1.1 in user_answers:
                lang_answer = user_answers[1.1]
                if lang_answer == 0: 
                    tags["Lang"].append("Python")
                elif lang_answer == 1: 
                    tags["Lang"].append("GO")
                elif lang_answer == 2: 
                    tags["Lang"].append("JavaScript")
                elif lang_answer == 3: 
                    tags["Lang"].extend(["C", "C++"])
                elif lang_answer == 4:  
                    tags["Lang"].append("Java")

    if 2 in user_answers:
        exp_answer = user_answers[2]
        if exp_answer == 0:  
            tags["Lv1"].append("Beginner")
        elif exp_answer == 1: 
            tags["Lv1"].extend(["Beginner", "Intermediate"])
        elif exp_answer == 2:  
            tags["Lv1"].append("Intermediate")
        elif exp_answer == 3: 
            tags["Lv1"].append("Expert")
    
    if 3 in user_answers:
        interest_answer = user_answers[3]
        if interest_answer == 0: 
            tags["Department"].extend(["Frontend", "Backend"])
        elif interest_answer == 1:  
            tags["Department"].extend(["Data Science", "Business Analytics"])
            
            if 3.1 in user_answers:
                data_answer = user_answers[3.1]
                if data_answer == 0: 
                    tags["Department"].append("Data Science")
                elif data_answer == 1:  
                    tags["Department"].append("Data Engineering")
                elif data_answer == 2:  
                    tags["Department"].append("Business Analytics")
                elif data_answer == 3:  
                    tags["Department"].append("Big Data")
                    
        elif interest_answer == 2: 
            tags["Department"].append("Cybersecurity")
        elif interest_answer == 3: 
            tags["Department"].append("UI/UX Design")
    
    if 4 in user_answers:
        project_answer = user_answers[4]
        if project_answer == 0:   

            if 4.1 in user_answers:
                platform_answer = user_answers[4.1]
                if platform_answer == 0:  
                    tags["Department"].append("iOS Development")
                elif platform_answer == 1: 
                    tags["Department"].append("Android Development")
                elif platform_answer == 2:  
                    tags["Department"].append("Cross-platform Development")      

        elif project_answer == 1:
            tags["Department"].append("Game Development")
        elif project_answer == 2: 
            tags["Department"].extend(["AI", "ML"])
        elif project_answer == 3: 
            tags["Department"].extend(["Frontend", "Backend"])
            
            if 4.1 in user_answers:
                web_answer = user_answers[4.1]
                if web_answer == 0:  
                    tags["Department"].append("Frontend")
                elif web_answer == 1: 
                    tags["Department"].append("Backend")
                elif web_answer == 2:  
                    tags["Department"].append("Fullstack")
    
    if 5 in user_answers:
        work_answer = user_answers[5]
        if work_answer == 0:              
            if 5.1 in user_answers:
                creative_answer = user_answers[5.1]
                if creative_answer == 0:  
                    tags["Department"].append("UI/UX Design")
                elif creative_answer == 1: 
                    tags["Department"].append("Game Development")
                    
        elif work_answer == 1:              
            if 5.1 in user_answers:
                logic_answer = user_answers[5.1]
                if logic_answer == 0:  
                    tags["Department"].append("Data Science")
                elif logic_answer == 1: 
                    tags["Department"].append("Software Development")
                elif logic_answer == 2: 
                    tags["Department"].extend(["Data Science", "Software Development"])
                    
        elif work_answer == 2:             
            if 5.1 in user_answers:
                org_answer = user_answers[5.1]
                if org_answer == 0: 
                    tags["Department"].append("Project Management")
                elif org_answer == 1: 
                    tags["Department"].append("Product Management")
                elif org_answer == 2: 
                    tags["Department"].extend(["Project Management", "Product Management"])
                    
        elif work_answer == 3: 
            tags["Department"].append("Cybersecurity")
    
    if 6 in user_answers:
        math_answer = user_answers[6]
        if math_answer == 0:   

            if 6.1 in user_answers:
                math_interest = user_answers[6.1]
                if math_interest == 0: 
                    tags["Department"].append("Data Science")
                elif math_interest == 1: 
                    tags["Department"].extend(["AI", "ML"])
                elif math_interest == 2:  
                    tags["Department"].append("Big Data")
                    
        elif math_answer == 1: 
            tags["Add tags"].extend(["Frontend", "Game Development"])
        elif math_answer == 2: 

            if 6.1 in user_answers:
                no_math_answer = user_answers[6.1]
                if no_math_answer == 0: 
                    tags["Department"].append("UI/UX Design")
                elif no_math_answer == 1: 
                    tags["Department"].append("Testing and QA")
                elif no_math_answer == 2:  
                    tags["Department"].append("Product Management")
    
    if 7 in user_answers:
        activity_answer = user_answers[7]
        if activity_answer == 0: 
            tags["Add tags"].extend(["Software Development", "Game Development"])
        elif activity_answer == 1:  
            tags["Add tags"].append("Testing and QA")
        elif activity_answer == 2:  
            
            if 7.1 in user_answers:
                optimize_answer = user_answers[7.1]
                if optimize_answer == 0:  
                    tags["Department"].append("DevOps")
                elif optimize_answer == 1:  
                    tags["Department"].append("System Administration")
                elif optimize_answer == 2:  
                    tags["Department"].append("DevOps")
                    
        elif activity_answer == 3:  
            tags["Add tags"].extend(["Product Management", "Project Management"])
    
    if 8 in user_answers:
        project_type = user_answers[8]
        if project_type == 0:
            tags["Department"].extend(["Backend", "DevOps"])
        elif project_type == 1:  
            tags["Department"].extend(["Frontend", "Fullstack"])
        elif project_type == 2: 
            tags["Department"].append("Game Development")
        elif project_type == 3: 
            tags["Department"].extend(["Data Science", "AI/ML"])
    
    for key in tags:
        tags[key] = list(set(tags[key]))
    
    return {"user_id": tags}  