import Main from "@components/Main/Main";
import MultiStepSurvey from "@components/MultiStepSurvey/MultiStepSurvey";
import { questions } from "@mocks/questions";

import "./RegistrationPage.css";

const RegistrationPage: React.FC = () => {
  return (
    <Main disableHeaderOffset>
      <MultiStepSurvey
        questions={questions}
        userMeta={JSON.parse(JSON.stringify({ name: "Alex" }))}
      />
    </Main>
  );
};

export default RegistrationPage;
