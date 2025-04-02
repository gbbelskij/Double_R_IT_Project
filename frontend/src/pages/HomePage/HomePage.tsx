import React from "react";
import Footer from "@components/Footer/Footer";
import Header from "@components/Header/Header";
import Logo from "@components/Logo/Logo";
import Main from "@components/Main/Main";
import CourseSection from "../../components/CourseSection/CourseSection"; // Импорт CourseSection
import { courses } from "../../data"; // Импорт данных курсов

import "./HomePage.css";

const HomePage: React.FC = () => {
  return (
    <>
      <Header />
      <Main>
        <Logo hasText />
        <CourseSection courses={courses} title="РЕКОМЕНДОВАННЫЕ ВАМ" />
      </Main>
      <Footer />
    </>
  );
};

export default HomePage;
