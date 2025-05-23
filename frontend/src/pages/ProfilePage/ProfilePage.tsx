import { useEffect, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useNavigate } from "react-router";
import { ScaleLoader } from "react-spinners";
import axios from "axios";

import { MdOutlineWorkOutline } from "react-icons/md";

import { ProfileFormData, profileSchema } from "@schemas/profileSchema";

import { useWindowSize } from "@hooks/useWindowSize";
import { useAuthGuard } from "@hooks/useAuthGuard";

import Footer from "@components/Footer/Footer";
import Header from "@components/Header/Header";
import Main from "@components/Main/Main";
import Input from "@components/Input/Input";
import Select from "@components/Select/Select";
import SmartPasswordInput from "@components/SmartPasswordInput/SmartPasswordInput";
import MultiStepSurvey from "@components/MultiStepSurvey/MultiStepSurvey";
import BackgroundElements from "@components/BackgroundElements/BackgroundElements";

import { SurveyData } from "@components/MultiStepSurvey/MultiStepSurvey.types";

import ProfileSurveyIntro from "./components/ProfileSurveyIntro/ProfileSurveyIntro";
import ProfileSurveyOutro from "./components/ProfileSurveyOutro/ProfileSurveyOutro";
import ProfileForm from "./components/ProfileForm/ProfileForm";

import { Question } from "types/question";

import { handleErrorNavigation } from "@utils/handleErrorNavigation";
import { declineYear } from "@utils/decline";

import { jobPositionOptions } from "@data/jobPositionOptions";

import "./ProfilePage.css";

const ProfilePage: React.FC = () => {
  const isChecking = useAuthGuard();

  const navigate = useNavigate();

  const [isLoading, setIsLoading] = useState(true);
  const [showSurvey, setShowSurvey] = useState(false);
  const [questionsLoading, setQuestionsLoading] = useState(false);
  const [questions, setQuestions] = useState<Record<string, Question>>({});

  const { isSmallMobile } = useWindowSize();

  const surveyRef = useRef(null);
  const formRef = useRef(null);

  const {
    register,
    handleSubmit,
    reset,
    watch,
    setError,
    formState: { errors, isValid, isSubmitted },
  } = useForm<ProfileFormData>({
    resolver: zodResolver(profileSchema),
    mode: "onSubmit",
    defaultValues: {
      work_experience: 0,
    },
  });

  useEffect(() => {
    if (isChecking) {
      return;
    }

    const fetchProfile = async () => {
      try {
        const response = await axios.get("/api/personal_account/", {
          withCredentials: true,
        });

        const data = response.data.user_data;

        reset({
          first_name: data.first_name,
          last_name: data.last_name,
          date_of_birth: data.date_of_birth,
          email: data.email,
          job_position: data.job_position,
          work_experience: data.work_experience,
          old_password: "",
          password: "",
          repeatPassword: "",
        });
      } catch (error) {
        handleErrorNavigation(
          error,
          navigate,
          "Ошибка загрузки профиля",
          "Не удалось получить данные пользователя."
        );
      } finally {
        setIsLoading(false);
      }
    };

    fetchProfile();
  }, [isChecking, reset, navigate]);

  useEffect(() => {
    if (Object.keys(questions).length > 0) {
      return;
    }

    const fetchQuestions = async () => {
      setQuestionsLoading(true);

      try {
        const response = await axios.get("api/register/questions/");

        setQuestions(response.data.questions);
      } catch (error) {
        handleErrorNavigation(error, navigate, "Ошибка загрузки опроса");
      } finally {
        setQuestionsLoading(false);
      }
    };

    fetchQuestions();
  }, [navigate, questions]);

  if (isChecking || isLoading || questionsLoading) {
    return (
      <Main disableHeaderOffset>
        <ScaleLoader color={"var(--solitude-100)"} />
      </Main>
    );
  }

  const isButtonDisabled =
    isSubmitted && (!isValid || Object.keys(errors).length > 0);

  const handleSurveyToggle = () => {
    setShowSurvey((prev) => !prev);
  };

  const handleSurveyExit = () => {
    setShowSurvey(false);
    reset();
  };

  const updateProfile = async (data: ProfileFormData) => {
    try {
      await axios.patch("/api/personal_account/update/", data, {
        withCredentials: true,
      });

      alert("Данные были успешно обновлены!");
    } catch (error) {
      let errorMessage = "Что-то пошло не так. Попробуйте позже.";

      if (axios.isAxiosError(error) && error.response?.data?.message) {
        errorMessage = error.response.data.message;

        if (errorMessage === "Old password is incorrect") {
          setError("old_password", {
            type: "manual",
            message: "Неверный пароль",
          });

          return;
        }
      }

      navigate("/error", {
        state: {
          errorHeading: "Ошибка обновления данных аккаунта",
          errorText: errorMessage,
          timeout: 0,
        },
      });
    }
  };

  const handleExit = async () => {
    try {
      await axios.post("/api/personal_account/logout/", null, {
        withCredentials: true,
      });

      document.cookie = "token=; Max-Age=0; path=/";

      navigate("/login");
    } catch (error) {
      handleErrorNavigation(error, navigate, "Ошибка выхода из аккаунта");
    }
  };

  const handleDelete = async () => {
    const confirmed = window.confirm("Вы уверены, что хотите удалить аккаунт?");

    if (!confirmed) {
      return;
    }

    try {
      await axios.delete("/api/personal_account/delete/", {
        withCredentials: true,
      });

      document.cookie = "token=; Max-Age=0; path=/";

      navigate("/login");
    } catch (error) {
      handleErrorNavigation(error, navigate, "Ошибка удаления аккаунта");
    }
  };

  const handleSurveyComplete = async (answers: SurveyData) => {
    try {
      await axios.patch(
        "/api/personal_account/update/",
        { preferences: answers },
        {
          withCredentials: true,
        }
      );

      alert("Данные были успешно обновлены!");
    } catch (error) {
      handleErrorNavigation(error, navigate, "Ошибка отправки ответов");
    }
  };

  return (
    <>
      <Header onProfileClick={handleSurveyExit} />
      <Main>
        {showSurvey ? (
          <MultiStepSurvey
            questions={questions}
            onComplete={handleSurveyComplete}
            onLogoClick={handleSurveyExit}
            onExit={handleSurveyExit}
            Intro={ProfileSurveyIntro}
            Outro={ProfileSurveyOutro}
            ref={surveyRef}
          />
        ) : (
          <ProfileForm
            handleAction={handleSubmit(updateProfile)}
            isButtonDisabled={isButtonDisabled}
            handleSurvey={handleSurveyToggle}
            handleExit={handleExit}
            handleDelete={handleDelete}
            ref={formRef}
          >
            <Input
              type="text"
              name="first_name"
              label="Имя"
              placeholder="Имя"
              register={register}
              error={errors.first_name}
            />
            <Input
              type="text"
              name="last_name"
              label="Фамилия"
              placeholder="Фамилия"
              register={register}
              error={errors.last_name}
            />
            <Input
              type="date"
              name="date_of_birth"
              label="Дата рождения"
              register={register}
              error={errors.date_of_birth}
            />
            <Input
              type="email"
              name="email"
              label="Эл. почта"
              register={register}
              error={errors.email}
            />
            <Select
              options={jobPositionOptions}
              name="job_position"
              label="Должность"
              placeholder="Выберите должность"
              icon={MdOutlineWorkOutline}
              register={register}
              error={errors.job_position}
              watch={watch}
            />
            <Input
              type="experience"
              name="work_experience"
              label="Опыт"
              defaultValue="0"
              getUnit={declineYear}
              register={register}
              error={errors.work_experience}
              valueAsNumber
            />
            <Input
              type="password"
              name="old_password"
              label="Старый пароль"
              register={register}
              error={errors.old_password}
            />
            <SmartPasswordInput
              label="Новый пароль"
              register={register}
              error={errors.password}
              repeatError={errors.repeatPassword}
            />
          </ProfileForm>
        )}

        <BackgroundElements
          targetRef={showSurvey ? surveyRef : formRef}
          blobsSize={showSurvey ? undefined : isSmallMobile ? 200 : 500}
          blobsRandomness={20}
          styles={showSurvey ? undefined : "profile-bg-elements"}
        />
      </Main>
      <Footer />
    </>
  );
};

export default ProfilePage;
