import React from "react";
import Footer from "@components/Footer/Footer";
import Header from "@components/Header/Header";
import Logo from "@components/Logo/Logo";
import Main from "@components/Main/Main";
import CourseSection from "../../components/CourseSection/CourseSection"; 
import { courses } from "../../data"; 

import "./HomePage.css";

const HomePage: React.FC = () => {
  return (
    <>
      <Header />
      <Main>
        
        <CourseSection courses={courses} title="РЕКОМЕНДОВАННЫЕ ВАМ" />
        <CourseSection courses={courses} title="ПОПУЛЯРНОЕ" />
      </Main>
      <Footer />
    </>
  );
};

export default HomePage;
