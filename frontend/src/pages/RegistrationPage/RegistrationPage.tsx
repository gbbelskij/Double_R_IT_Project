import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  registrationSchema,
  RegistrationFormData,
} from "@schemas/registrationSchema";
import axios from "axios";
import { ScaleLoader } from "react-spinners";

import { MdOutlineWorkOutline } from "react-icons/md";

import Main from "@components/Main/Main";
import Form from "@components/Form/Form";
import Input from "@components/Input/Input";
import SmartPasswordInput from "@components/SmartPasswordInput/SmartPasswordInput";
import Select from "@components/Select/Select";
import MultiStepSurvey from "@components/MultiStepSurvey/MultiStepSurvey";
import BackgroundElements from "@components/BackgroundElements/BackgroundElements";

import { SurveyData } from "@components/MultiStepSurvey/MultiStepSurvey.types";

import { Question } from "types/question";

import { declineYear } from "@utils/decline";
import { handleErrorNavigation } from "@utils/handleErrorNavigation";

import { useWindowSize } from "@hooks/useWindowSize";

import { jobPositionOptions } from "@data/jobPositionOptions";

import "./RegistrationPage.css";

const RegistrationPage: React.FC = () => {
  const [regStep, setRegStep] = useState(1);
  const [questions, setQuestions] = useState<Record<string, Question>>({});
  const [questionsLoading, setQuestionsLoading] = useState(false);
  const [registrationLoading, setRegistrationLoading] = useState(false);
  const [regFirstStepInfo, setRegFirstStepInfo] =
    useState<RegistrationFormData | null>(null);

  const navigate = useNavigate();

  const { isMobile, isSmallMobile } = useWindowSize();

  const sectionRef = useRef(null);

  useEffect(() => {
    if (Object.keys(questions).length > 0) {
      return;
    }

    const fetchQuestions = async () => {
      setQuestionsLoading(true);

      try {
        const response = await axios.get("api/register/questions");

        setQuestions(response.data.questions);
      } catch (error) {
        handleErrorNavigation(error, navigate, "Ошибка загрузки опроса");
      } finally {
        setQuestionsLoading(false);
      }
    };

    fetchQuestions();
  }, [regStep, navigate, questions]);

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitted, isValid },
  } = useForm<RegistrationFormData>({
    resolver: zodResolver(registrationSchema),
    mode: "onSubmit",
    defaultValues: {
      work_experience: 0,
    },
  });

  const finishFirstStep = (data: RegistrationFormData) => {
    setRegFirstStepInfo(data);
    setRegStep(2);
  };

  const submitFullRegistration = async (
    answers: SurveyData,
    userMeta: RegistrationFormData
  ) => {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { repeatPassword, ...formDataToSend } = userMeta;

    const data = {
      preferences: answers,
      ...formDataToSend,
    };

    setRegistrationLoading(true);

    try {
      await axios.post("api/register", data);
    } catch (error) {
      handleErrorNavigation(
        error,
        navigate,
        "Ошибка регистрации",
        "Что-то пошло не так. Попробуйте позже."
      );
    } finally {
      setRegistrationLoading(false);
    }
  };

  const handleSurveyExit = () => {
    setRegStep(1);
  };

  const isButtonDisabled =
    isSubmitted && (!isValid || Object.keys(errors).length > 0);

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
          isButtonDisabled={isButtonDisabled}
          logoOffset={isSmallMobile ? 30 : isMobile ? 40 : 50}
          logoAlign="end"
          ref={sectionRef}
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
          <SmartPasswordInput
            register={register}
            error={errors.password}
            repeatError={errors.repeatPassword}
          />
        </Form>
      )}

      {regStep === 2 && (
        <>
          {questionsLoading ? (
            <ScaleLoader color={"var(--solitude-100)"} />
          ) : (
            <>
              <MultiStepSurvey
                questions={questions}
                userMeta={regFirstStepInfo}
                onComplete={submitFullRegistration}
                onLogoClick={handleSurveyExit}
                loading={registrationLoading}
                ref={sectionRef}
              />

              {!registrationLoading && (
                <BackgroundElements targetRef={sectionRef} />
              )}
            </>
          )}
        </>
      )}
    </Main>
  );
};

export default RegistrationPage;
