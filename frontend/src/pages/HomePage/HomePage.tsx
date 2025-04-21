import Footer from "@components/Footer/Footer";
import Header from "@components/Header/Header";
import Main from "@components/Main/Main";
import CourseSection from "@components/CourseSection/CourseSection";
import { courses } from "@mocks/courses";

import "./HomePage.css";

const HomePage: React.FC = () => {
  return (
    <>
      <Header />
      <Main>
        <CourseSection courses={courses} title="Рекомендованные вам" />
        <CourseSection courses={courses} title="Популярное" />
      </Main>
      <Footer />
    </>
  );
};

export default HomePage;
