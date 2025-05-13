import { useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  registrationSchema,
  RegistrationFormData,
} from "@schemas/registrationSchema";
import axios from "axios";

import { MdOutlineWorkOutline } from "react-icons/md";

import { declineYear } from "@utils/decline";

import { useWindowSize } from "@hooks/useWindowSize";

import Main from "@components/Main/Main";
import Form from "@components/Form/Form";
import Input from "@components/Input/Input";
import SmartPasswordInput from "@components/SmartPasswordInput/SmartPasswordInput";
import Select from "@components/Select/Select";
import MultiStepSurvey from "@components/MultiStepSurvey/MultiStepSurvey";
import BackgroundElements from "@components/BackgroundElements/BackgroundElements";

import { SurveyData } from "@components/MultiStepSurvey/MultiStepSurvey.types";

import { questions } from "@mocks/questions";
import { options } from "@mocks/options";

import "./RegistrationPage.css";

const RegistrationPage: React.FC = () => {
  const [regStep, setRegStep] = useState(1);
  const [regFirstStepInfo, setRegFirstStepInfo] =
    useState<RegistrationFormData | null>(null);

  const { isMobile, isSmallMobile } = useWindowSize();

  const sectionRef = useRef(null);

  const {
    register,
    handleSubmit,
    formState: { errors, isValid, isSubmitted },
  } = useForm<RegistrationFormData>({
    resolver: zodResolver(registrationSchema),
    mode: "onSubmit",
    defaultValues: {
      experience: 0,
    },
  });

  const finishFirstStep = (data: RegistrationFormData) => {
    setRegFirstStepInfo(data);
    setRegStep(2);
  };

  const isButtonDisabled =
    isSubmitted && (!isValid || Object.keys(errors).length > 0);

  const submitFullRegistration = async (
    answers: SurveyData,
    userMeta: RegistrationFormData
  ) => {
    try {
      await axios.post("/api/registration", {
        answers: answers,
        userMeta: userMeta,
      });
      console.log("Registration successful");
    } catch (error) {
      console.error("Registration failed:", error);
    }
  };

  return (
    <Main disableHeaderOffset>
      {regStep === 1 && (
        <Form
          id="registration-form"
          formClassName="registration-form"
          title="Регистрация"
          additionalText="Давайте познакомимся!"
          actionText="Далее"
          helperText="У вас уже есть аккаунт?"
          helperLink="/login"
          helperLinkText="Войти"
          handleAction={handleSubmit(finishFirstStep)}
          logoOffset={isSmallMobile ? 30 : isMobile ? 40 : 50}
          logoAlign="end"
          isButtonDisabled={isButtonDisabled}
          ref={sectionRef}
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
            defaultValue="0"
            getUnit={declineYear}
            register={register}
            error={errors.experience}
            valueAsNumber
          />
          <SmartPasswordInput
            register={register}
            error={errors.password}
            repeatError={errors.repeatPassword}
          />
        </Form>
      )}

      {regStep === 2 && (
        <MultiStepSurvey
          questions={questions}
          userMeta={regFirstStepInfo}
          onComplete={submitFullRegistration}
          ref={sectionRef}
        />
      )}

      <BackgroundElements targetRef={sectionRef} />
    </Main>
  );
};

export default RegistrationPage;
