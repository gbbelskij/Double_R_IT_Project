import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useNavigate } from "react-router";
import axios from "axios";

import { MdOutlineWorkOutline } from "react-icons/md";

import { ProfileFormData, profileSchema } from "@schemas/profileSchema";

import Footer from "@components/Footer/Footer";
import Header from "@components/Header/Header";
import Main from "@components/Main/Main";
import Input from "@components/Input/Input";
import Select from "@components/Select/Select";
import SmartPasswordInput from "@components/SmartPasswordInput/SmartPasswordInput";
import MultiStepSurvey from "@components/MultiStepSurvey/MultiStepSurvey";
import ProfileSurveyIntro from "./components/ProfileSurveyIntro/ProfileSurveyIntro";
import ProfileSurveyOutro from "./components/ProfileSurveyOutro/ProfileSurveyOutro";
import BackgroundElements from "@components/BackgroundElements/BackgroundElements";

import ProfileForm from "./components/ProfileForm/ProfileForm";

import { SurveyData } from "@components/MultiStepSurvey/MultiStepSurvey.types";

import { questions } from "@mocks/questions";
import { options } from "@mocks/options";
import { declineYear } from "@utils/decline";

import "./ProfilePage.css";

const ProfilePage: React.FC = () => {
  const [showSurvey, setShowSurvey] = useState(false);
  const [surveyComplete, setSurveyComplete] = useState(false);
  const navigate = useNavigate();

  const {
    register,
    handleSubmit,
    formState: { errors, isValid, isSubmitted },
  } = useForm<ProfileFormData>({
    resolver: zodResolver(profileSchema),
    mode: "onSubmit",
  });

  const isButtonDisabled =
    isSubmitted && (!isValid || Object.keys(errors).length > 0);

  const updateProfile = async (data: ProfileFormData) => {
    try {
      await axios.put("/api/profile", data);
      console.log("Profile updated");
    } catch (error) {
      console.error("Update failed:", error);
    }
  };

  const handleSurveyToggle = () => {
    setShowSurvey((prev) => !prev);
  };

  const handleExit = async () => {
    try {
      await axios.post("/api/logout");
      navigate("/login");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  const handleDelete = async () => {
    const confirmed = window.confirm("Вы уверены, что хотите удалить аккаунт?");
    if (!confirmed) return;

    try {
      await axios.delete("/api/profile");
      navigate("/goodbye");
    } catch (error) {
      console.error("Delete failed:", error);
    }
  };

  const handleSurveyComplete = async (answers: SurveyData) => {
    try {
      await axios.post("/api/profile/survey", { answers });
      setSurveyComplete(true);
      setShowSurvey(false);
    } catch (error) {
      console.error("Survey submit failed:", error);
    }
  };

  return (
    <>
      <Header />
      <Main>
        {showSurvey && !surveyComplete ? (
          <MultiStepSurvey
            questions={questions}
            onComplete={handleSurveyComplete}
            Intro={ProfileSurveyIntro}
            Outro={ProfileSurveyOutro}
          />
        ) : (
          <ProfileForm
            handleAction={handleSubmit(updateProfile)}
            isButtonDisabled={isButtonDisabled}
            handleSurvey={handleSurveyToggle}
            handleExit={handleExit}
            handleDelete={handleDelete}
          >
            <Input
              type="text"
              name="name"
              label="Имя"
              placeholder="Имя"
              register={register}
              error={errors.name}
            />
            <Input
              type="text"
              name="surname"
              label="Фамилия"
              placeholder="Фамилия"
              register={register}
              error={errors.surname}
            />
            <Input
              type="date"
              name="birthday"
              label="Дата рождения"
              register={register}
              error={errors.birthday}
            />
            <Input
              type="email"
              name="email"
              label="Эл. почта"
              register={register}
              error={errors.email}
            />
            <Select
              options={options}
              name="post"
              label="Должность"
              placeholder="Выберите должность"
              icon={MdOutlineWorkOutline}
              register={register}
              error={errors.post}
            />
            <Input
              type="experience"
              name="experience"
              label="Опыт"
              getUnit={declineYear}
              register={register}
              error={errors.experience}
              valueAsNumber
            />
            <Input
              type="password"
              name="oldPassword"
              label="Старый пароль"
              register={register}
              error={errors.oldPassword}
              valueAsNumber
            />
            <SmartPasswordInput
              label="Новый пароль"
              register={register}
              error={errors.password}
              repeatError={errors.repeatPassword}
            />

            <BackgroundElements />
          </ProfileForm>
        )}
      </Main>
      <Footer />
    </>
  );
};

export default ProfilePage;
