class DemoController < ApplicationController

  def new
  end

  def create
    result = `cd ./lib/ML && python3.6 #{Rails.root}/lib/ML/Hackana_beta.py "#{params[:carater_atendimento]};#{params[:service_id]};#{params[:sexo]};#{params[:item_kind]};#{params[:charged_value]};#{params[:days_inside]};#{params[:years_old]}"`
    result.gsub!("\n","")
    if result == "1"
      redirect_to action: :failure and return    
    else
      redirect_to action: :success and return    
    end
  end

  def failure
  end

  def success
  end
end
