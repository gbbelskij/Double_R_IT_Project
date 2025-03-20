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
        <Logo hasText />
        <CourseSection courses={courses} title="Our Courses" /> {/* Добавлен CourseSection */}
      </Main>
      <Footer />
    </>
  );
};

export default HomePage;
