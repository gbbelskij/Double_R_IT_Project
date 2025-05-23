import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { ScaleLoader } from "react-spinners";
import axios from "axios";

import Footer from "@components/Footer/Footer";
import Header from "@components/Header/Header";
import Main from "@components/Main/Main";
import CourseSection from "@components/CourseSection/CourseSection";

import { useAuthGuard } from "@hooks/useAuthGuard";

import { handleErrorNavigation } from "@utils/handleErrorNavigation";
import { normalizeCourses } from "@utils/normalizeCourses";

import { Course } from "types/course";

const HomePage: React.FC = () => {
  const isChecking = useAuthGuard();

  const navigate = useNavigate();

  const [recommendedCourses, setRecommendedCourses] = useState<Course[]>([]);
  const [popularCourses, setPopularCourses] = useState<Course[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (isChecking) {
      return;
    }

    const fetchCourses = async () => {
      try {
        const [recommendedRes, popularRes] = await Promise.all([
          axios.get("/api/mainpage/recommended_cources/", {
            withCredentials: true,
          }),
          axios.get("/api/mainpage/all_cources/", {
            withCredentials: true,
          }),
        ]);

        setRecommendedCourses(normalizeCourses(recommendedRes.data.courses));
        setPopularCourses(normalizeCourses(popularRes.data.courses));
      } catch (error) {
        handleErrorNavigation(error, navigate, "Ошибка загрузки курсов");
      } finally {
        setIsLoading(false);
      }
    };

    fetchCourses();
  }, [isChecking, navigate]);

  if (isChecking || isLoading) {
    return (
      <Main disableHeaderOffset>
        <ScaleLoader color={"var(--solitude-100)"} />
      </Main>
    );
  }

  return (
    <>
      <Header />
      <Main>
        <CourseSection
          courses={recommendedCourses}
          title="Рекомендованные вам"
        />
        <CourseSection courses={popularCourses} title="Популярное" />
      </Main>
      <Footer />
    </>
  );
};

export default HomePage;
